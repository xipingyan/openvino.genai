// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <openvino_extensions/strings.hpp>
#include <valarray>

namespace {
std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest&& tokenizer, std::string_view prompt) {
    constexpr size_t BATCH_SIZE = 1;
    ov::Tensor destination = tokenizer.get_input_tensor();
    openvino_extensions::pack_strings(std::array<std::string_view, BATCH_SIZE>{prompt}, destination);
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

void print_token(ov::InferRequest& detokenizer, int64_t out_token) {
    constexpr size_t BATCH_SIZE = 1;
    ov::Tensor inp = detokenizer.get_input_tensor();
    inp.set_shape({BATCH_SIZE, 1});
    inp.data<int64_t>()[0] = out_token;
    detokenizer.infer();
    std::cout << openvino_extensions::unpack_strings(detokenizer.get_output_tensor()).front() << std::flush;
}

// Modifyed Knuth–Morris–Pratt algorithm which returns a set of tokens following after every needle occurance in haystack
std::vector<int64_t> kmp_search(const std::vector<int64_t>& haystack, std::vector<int64_t> needle) {  // TODO: pass iters to haystack to avoid searchng last ngram symbols
    if (needle.empty()) {  // NO_REPEAT_NGRAM_SIZE == 1, ban every symbol
        return {haystack.begin(), haystack.end()};
    }
    std::vector<int> partial_match_table(needle.size() + 1, -1);
    int cnd = 0;
    for (size_t pos = 1; pos < needle.size(); ++pos) {
        if (needle[pos] == needle[cnd]) {
            partial_match_table[pos] = partial_match_table[cnd];
        } else {
            partial_match_table[pos] = cnd;
            while (cnd >= 0 && needle[pos] != needle[cnd]) {
                cnd = partial_match_table[cnd];
            }
        }
        ++cnd;
    }
    partial_match_table.back() = cnd;
    std::vector<int64_t> res;
    size_t j = 0;  // The position of the current character in haystack
    int k = 0;  // The position of the current character in needle
    while (j < haystack.size() - 1) {
        if (needle[k] == haystack[j]) {
            ++j;
            ++k;
            if (k == int(needle.size())) {
                res.push_back(haystack[j]);
                k = partial_match_table[k];
            }
        } else {
            k = partial_match_table[k];
            if (k < 0) {
                ++j;
                ++k;
            }
        }
    }
    return res;
}
enum class StopCriteria {early, heuristic, never};
size_t MAX_NEW_TOKENS;
size_t N_GROUPS;
size_t GROUP_SIZE;
StopCriteria stop_criteria;
size_t NO_REPEAT_NGRAM_SIZE;
float DIVERSITY_PENALTY;
double LENGTH_PENALTY;  // TODO: align defaults with transformers
int64_t EOS_TOKEN;  // There's no way to extract the value from the tokenizer for now  // TODO: 2 for llama2
}

    struct Beam {
        float log_prob;
        std::vector<int64_t> tokens;
        size_t batch_id = 0;
        bool operator<(const Beam& other) {
            return log_prob > other.log_prob;  // greater, not less to build min heap
        }
        Beam() : log_prob{-1e9} {}
        Beam& operator=(Beam&& other) {
            log_prob = other.log_prob;
            tokens = std::move(other.tokens);
            batch_id = other.batch_id;
            return *this;
        }
        Beam& operator=(const Beam& other) {
            log_prob = other.log_prob;
            tokens = other.tokens;
            batch_id = other.batch_id;
            return *this;
        }
        Beam(Beam&& other) : log_prob{other.log_prob}, tokens{std::move(other.tokens)}, batch_id{other.batch_id} {};
        Beam(const Beam& other) : log_prob{other.log_prob}, tokens{other.tokens}, batch_id{other.batch_id} {};
    };

std::ostream& operator<<(std::ostream& os, const Beam& beam) {
    os << std::setprecision(6) << beam.log_prob << ": ";
    for (size_t token : beam.tokens) {
        os << token << ' ';
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<Beam>& beams) {
    for (const Beam& beam : beams) {
        os << beam << '\n';
    }
    return os;
}


struct Hypotheses {
        std::vector<Beam> beams;
        bool done = false;
        void push(Beam&& beam, size_t prompt_light) {
            beam.log_prob = double(beam.log_prob) / std::pow(beam.tokens.size() + prompt_light, LENGTH_PENALTY);
            beams.push_back(std::move(beam));
            std::push_heap(beams.begin(), beams.end());
            if (beams.size() > GROUP_SIZE) {
                std::pop_heap(beams.begin(), beams.end());
                beams.pop_back();
            }
        }
        bool is_done(double best_sum_logprobs, size_t cur_len) {   // TODO: just done()?
            if (beams.size() < GROUP_SIZE) {
                return false;
            }
            switch (stop_criteria) {
                case StopCriteria::early: done = true; return true;
                case StopCriteria::heuristic: {
                    double worst_score = beams.front().log_prob;
                    double highest_attainable_score = best_sum_logprobs / std::pow(double(cur_len), LENGTH_PENALTY);
                    done = worst_score >= highest_attainable_score;
                    return worst_score >= highest_attainable_score;
                }
                case StopCriteria::never: {
                    double worst_score = beams.front().log_prob;
                    double highest_attainable_score = LENGTH_PENALTY > 0.0f ? best_sum_logprobs / std::pow(double(MAX_NEW_TOKENS), LENGTH_PENALTY) : best_sum_logprobs / std::pow(double(cur_len), LENGTH_PENALTY);
                    done = worst_score >= highest_attainable_score;
                    return worst_score >= highest_attainable_score;
                }
                default: throw std::runtime_error("Never reached");
            }
        }
    };
        struct Group {
        std::vector<Beam> beams;  // TODO: one contigous array with all beams?
        Hypotheses hypotheses;
    };

int main(int argc, char* argv[]) try {
    if (argc != 12) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <openvino_model.xml> <tokenizer.xml> <detokenizer.xml> '<prompt>'");
    }
    MAX_NEW_TOKENS = std::stoi(argv[5]);
    N_GROUPS = std::stoi(argv[6]);
    GROUP_SIZE = std::stoi(argv[7]);
    if (std::string{"early"} == argv[8]) {
        stop_criteria = StopCriteria::early;
    } else if (std::string{"heuristic"} == argv[8]) {
        stop_criteria = StopCriteria::heuristic;
    } else if (std::string{"never"} == argv[8]) {
        stop_criteria = StopCriteria::never;
    } else {
        throw std::runtime_error("Unknown stop_criteria value");
    }
    NO_REPEAT_NGRAM_SIZE = std::stoi(argv[9]);
    DIVERSITY_PENALTY = std::stof(argv[10]);
    LENGTH_PENALTY = std::stof(argv[11]);
    EOS_TOKEN = 2;
    ov::Core core;
    core.add_extension(USER_OV_EXTENSIONS_PATH);  // USER_OV_EXTENSIONS_PATH is defined in root CMakeLists.txt
    auto [input_ids, attention_mask] = tokenize(core.compile_model(argv[2], "CPU").create_infer_request(), argv[4]);
    ov::InferRequest detokenizer = core.compile_model(argv[3], "CPU").create_infer_request();
    size_t prompt_length = input_ids.get_size();
    std::shared_ptr<ov::Model> model = core.read_model(argv[1]);
    constexpr size_t BATCH_SIZE = 1;
    std::map<size_t, ov::PartialShape> shapes = {
        {0, ov::PartialShape{
            -1, -1
        }},
        {1, ov::PartialShape{
            -1, -1
        }},
        {2, ov::PartialShape{
            -1, -1
        }}
    };
    std::vector<ov::Output<ov::Node>> inputs = model->inputs();
    for (size_t idx = 3; idx < inputs.size(); ++idx) {
        ov::PartialShape shape = inputs.at(idx).get_partial_shape();
        shape[0] = -1;
        shapes.emplace(idx, shape);
    }
    model->reshape(shapes);
    ov::CompiledModel compiled = core.compile_model(model, "CPU");  // , ov::cache_dir("llm-cache"));

    struct Token {double log; int64_t idx;
        bool operator<(Token indexed) {
            return log > indexed.log;  // greater, not less to pick most probable tokens
        }
    };
    std::vector<Group> groups{N_GROUPS};
        ov::InferRequest ireq = compiled.create_infer_request();
        for (size_t idx = 3; idx < inputs.size(); ++idx) {
            ov::Shape shape = inputs.at(idx).get_partial_shape().get_min_shape();
            shape[0] = 1;
            ireq.get_input_tensor(idx).set_shape(shape);
        }
        ireq.get_tensor("input_ids").set_shape(input_ids.get_shape());  // TODO: replace with ireq.set_tensor("input_ids", input_ids); after it's fixed
        ireq.get_tensor("attention_mask").set_shape({BATCH_SIZE, ireq.get_tensor("input_ids").get_size()});
        std::copy_n(input_ids.data<const int64_t>(), input_ids.get_size(), ireq.get_tensor("input_ids").data<int64_t>());
        std::fill_n(ireq.get_tensor("attention_mask").data<int64_t>(), input_ids.get_size(), 1);
        ireq.get_tensor("position_ids").set_shape(input_ids.get_shape());
        std::iota(ireq.get_tensor("position_ids").data<int64_t>(), ireq.get_tensor("position_ids").data<int64_t>() + ireq.get_tensor("position_ids").get_size(), 0);
        ireq.infer();

    for (Group & group : groups) {
        group.beams.resize(GROUP_SIZE);
        group.beams.front().log_prob = 0.0;
    }
    size_t incomplete_groups = N_GROUPS;
    for (size_t length_count = 0; length_count < MAX_NEW_TOKENS; ++length_count) {
        for (size_t group_idx = 0; group_idx < incomplete_groups; ++group_idx) {
            if (groups[group_idx].hypotheses.done) {
                continue;
            }
            std::vector<Beam> candidates;
            candidates.reserve(2 * GROUP_SIZE);
            for (size_t beam_idx = 0; beam_idx < GROUP_SIZE; ++beam_idx) {
                ov::Tensor logits_tensor = ireq.get_tensor("logits");
                size_t vocab_size = logits_tensor.get_shape().back();
                std::vector<float> temp;
                size_t batch_offset = groups[group_idx].beams[beam_idx].batch_id * logits_tensor.get_shape()[1] * logits_tensor.get_shape()[2];
                for (size_t logit_id = 0; logit_id < vocab_size; ++logit_id) {
                    temp.push_back((logits_tensor.data<const float>() + batch_offset + (logits_tensor.get_shape()[1] - 1) * vocab_size)[logit_id]);
                }
                std::valarray<float> logits(temp.data(), temp.size());  // TODO: maybe use valarray<Token>
                float max_logit = logits.max();
                float log_sum = std::log((std::exp(logits - max_logit)).sum());  // TODO: log(softmax) only for topk logits
                std::valarray<float> log_prob = logits - max_logit - log_sum;
                std::vector<Token> tokens;
                tokens.reserve(log_prob.size());
                for (size_t idx = 0; idx < log_prob.size(); ++idx) {
                    tokens.push_back({log_prob[idx], int64_t(idx)});
                }
                for (size_t prev_group_idx = 0; prev_group_idx < group_idx; ++prev_group_idx) {  // TODO: range based for
                    for (size_t prev_beam_idx = 0; prev_beam_idx < GROUP_SIZE; ++prev_beam_idx) {
                        tokens[groups[prev_group_idx].beams[prev_beam_idx].tokens.back()].log -= DIVERSITY_PENALTY;
                    }
                }
                std::vector<int64_t>& other_tokens = groups[group_idx].beams[beam_idx].tokens;
                std::vector<int64_t> full_text;
                for (size_t idx = 0; idx < input_ids.get_size(); ++idx) {
                    full_text.push_back(input_ids.data<int64_t>()[idx]);
                }
                full_text.insert(full_text.end(), other_tokens.begin(), other_tokens.end());
                if (full_text.size() > 1 && full_text.size() >= NO_REPEAT_NGRAM_SIZE) {
                    for (int64_t ban_id : kmp_search(full_text, {full_text.end() - NO_REPEAT_NGRAM_SIZE + 1, full_text.end()})) {
                        tokens[ban_id].log = -std::numeric_limits<float>::infinity();
                    }
                }
                // Sample 2 * GROUP_SIZE next tokens to get at least 1 non EOS token per beam
                std::nth_element(tokens.begin(), tokens.begin() + 2 * GROUP_SIZE, tokens.end());
                for (size_t idx = 0; idx < 2 * GROUP_SIZE; ++idx) {
                    candidates.push_back(groups[group_idx].beams[beam_idx]);
                    candidates.back().log_prob += tokens[idx].log;
                    candidates.back().tokens.push_back(tokens[idx].idx);
                }
            }
            std::sort(candidates.begin(), candidates.end());  // TODO not sort
            size_t cur_len = groups[group_idx].beams.front().tokens.size() + 1;
            groups[group_idx].beams.clear();
            for (size_t cand_id = 0; cand_id < candidates.size(); ++cand_id) {
                if (EOS_TOKEN == candidates[cand_id].tokens.back()) {  // TODO: idx->token_id
                    // if beam_token does not belong to top num_beams tokens, it should not be added
                    if (cand_id >= GROUP_SIZE) {
                        continue;
                    }
                    candidates[cand_id].tokens.resize(candidates[cand_id].tokens.size() - 1);
                    groups[group_idx].hypotheses.push(std::move(candidates[cand_id]), prompt_length);
                } else {
                    groups[group_idx].beams.push_back(std::move(candidates[cand_id]));
                    if (groups[group_idx].beams.size() == GROUP_SIZE) {
                        break;
                    }
                }
            }
            groups[group_idx].hypotheses.is_done(cur_len + prompt_length, groups[group_idx].beams.front().log_prob);  // TODO: that requires groups[group_idx].beams to be not empty
        }
        size_t current_incomplete_groups = 0;
        for (Group& group : groups) {
            if (!group.hypotheses.done) {
                ++current_incomplete_groups;
            }
        }
        if (0 == current_incomplete_groups) {
            break;
        }
        size_t future_batch_size = 0;
        for (Group& group : groups) {
            if (group.hypotheses.done) {
                continue;
            }
            for (size_t beam_idx = 0; beam_idx < group.beams.size(); ++beam_idx) {
                ++future_batch_size;
            }
        }
        ireq.get_tensor("input_ids").set_shape({future_batch_size, 1});
        ov::Shape att_shape = ireq.get_tensor("attention_mask").get_shape();
        att_shape[0] = future_batch_size;
        ++att_shape[1];
        ireq.get_tensor("attention_mask").set_shape(att_shape);
        ireq.get_tensor("position_ids").set_shape({future_batch_size, 1});
        for (size_t tensor_id = 3; tensor_id < inputs.size(); ++tensor_id) {
            ov::Tensor past = ireq.get_input_tensor(tensor_id);
            ov::Shape shape = ireq.get_output_tensor(tensor_id - 2).get_shape();
            shape[0] = future_batch_size;
            past.set_shape(shape);
        }

        size_t batch_idx = 0;
        for (Group& group : groups) {
            if (group.hypotheses.done) {
                continue;
            }
            for (Beam& beam : group.beams) {
                ireq.get_tensor("input_ids").data<int64_t>()[batch_idx] = beam.tokens.back();
                std::fill_n(ireq.get_tensor("attention_mask").data<int64_t>(), ireq.get_tensor("attention_mask").get_size(), 1);
                ireq.get_tensor("position_ids").data<int64_t>()[batch_idx] = prompt_length + beam.tokens.size() - 1;
                for (size_t tensor_id = 3; tensor_id < inputs.size(); ++tensor_id) {
                    ov::Tensor present = ireq.get_output_tensor(tensor_id - 2);
                    ov::Shape present_begin = {beam.batch_id, 0, 0, 0};
                    ov::Shape present_end = present.get_shape();
                    present_end[0] = beam.batch_id + 1;
                    ov::Tensor past = ireq.get_input_tensor(tensor_id);
                    ov::Shape past_begin = {batch_idx, 0, 0, 0};
                    ov::Shape past_end = past.get_shape();
                    past_end[0] = batch_idx + 1;
                    ov::Tensor{present, present_begin, present_end}.copy_to(ov::Tensor{past, past_begin, past_end});
                }
                beam.batch_id = batch_idx;
                ++batch_idx;
            }
        }
        ireq.infer();
    }
    // finalize
    for (Group& group : groups) {
        if (group.hypotheses.is_done(group.beams.front().tokens.size() + prompt_length, group.beams.front().log_prob)) {
            continue;
        }
        for (Beam& beam: group.beams) {  // TODO: &&
            group.hypotheses.push(std::move(beam), prompt_length);
        }
    }
    for (Group& group: groups) {
        std::cout << "\nGroup:";
        std::sort_heap(group.hypotheses.beams.begin(), group.hypotheses.beams.end());
        for (const Beam& beam: group.hypotheses.beams) {
            std::cout << "\nscore: " << beam.log_prob << " prediction: ";  // TODO: alight with transformers
            for (int64_t token : beam.tokens) {
                print_token(detokenizer, token);
            }
        }
    }
    std::cout << '\n';
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
