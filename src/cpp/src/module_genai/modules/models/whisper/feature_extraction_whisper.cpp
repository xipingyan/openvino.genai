// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module_genai/modules/models/whisper/feature_extraction_whisper.hpp"

#include <algorithm>
#include <cstring>

#include <openvino/core/except.hpp>

namespace ov::genai::module {

WhisperFeatureExtractor::WhisperFeatureExtractor(const std::filesystem::path& model_path)
	: m_impl(model_path / "preprocessor_config.json") {}

WhisperFeatureExtractorOutput WhisperFeatureExtractor::extract(const std::vector<float>& raw_speech,
															  std::optional<size_t> sampling_rate,
															  bool return_attention_mask) {
	if (sampling_rate.has_value() && sampling_rate.value() != m_impl.sampling_rate) {
		OPENVINO_THROW("WhisperFeatureExtractor: expected sampling_rate=",
					  m_impl.sampling_rate,
					  ", got ",
					  sampling_rate.value());
	}

	const size_t target_samples = m_impl.n_samples;
	const size_t valid_samples = std::min(raw_speech.size(), target_samples);

	std::vector<float> padded;
	padded.reserve(target_samples);
	padded.insert(padded.end(), raw_speech.begin(), raw_speech.begin() + valid_samples);
	if (padded.size() < target_samples) {
		padded.resize(target_samples, 0.0f);
	}

	ov::genai::WhisperFeatures features = m_impl.extract(padded);

	ov::Tensor input_features(ov::element::f32, ov::Shape{1, features.feature_size, features.n_frames});
	std::memcpy(input_features.data(),
				features.data.data(),
				features.data.size() * sizeof(float));

	WhisperFeatureExtractorOutput out{std::move(input_features), std::nullopt, 0};

	// Frame count corresponding to the unpadded prefix.
	// Whisper STFT produces (L // hop_length) frames after dropping the last frame.
	out.num_frames = (m_impl.hop_length == 0) ? 0 : (valid_samples / m_impl.hop_length);
	out.num_frames = std::min(out.num_frames, static_cast<size_t>(features.n_frames));

	if (return_attention_mask) {
		ov::Tensor mask(ov::element::i32, ov::Shape{1, features.n_frames});
		auto* mask_data = mask.data<int32_t>();
		std::fill(mask_data, mask_data + features.n_frames, 0);
		std::fill(mask_data, mask_data + out.num_frames, 1);
		out.attention_mask = std::move(mask);
	}

	return out;
}

size_t WhisperFeatureExtractor::feature_size() const noexcept {
	return m_impl.feature_size;
}

size_t WhisperFeatureExtractor::sampling_rate() const noexcept {
	return m_impl.sampling_rate;
}

size_t WhisperFeatureExtractor::hop_length() const noexcept {
	return m_impl.hop_length;
}

size_t WhisperFeatureExtractor::n_fft() const noexcept {
	return m_impl.n_fft;
}

size_t WhisperFeatureExtractor::n_samples() const noexcept {
	return m_impl.n_samples;
}


} // namespace ov::genai::module
