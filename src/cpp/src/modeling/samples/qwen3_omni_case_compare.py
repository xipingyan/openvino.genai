#!/usr/bin/env python3

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path


def ensure_test_wav(path: Path, sample_rate: int = 16000, seconds: float = 1.0, freq_hz: float = 440.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(sample_rate * seconds)
    amplitude = 0.25
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for i in range(num_samples):
            sample = amplitude * math.sin(2.0 * math.pi * freq_hz * (i / sample_rate))
            value = int(max(-1.0, min(1.0, sample)) * 32767)
            frames.extend(value.to_bytes(2, byteorder="little", signed=True))
        wav.writeframes(bytes(frames))


def parse_cpp_text(stdout: str) -> str:
    """Extract model text output from C++ stdout.

    The C++ binary prints stats lines (containing ':') before the generated text,
    and '[SafetensorsWeightFinalizer]' lines after it.  We capture everything
    between the last stats line (Throughput) and the first '[' bracket line that
    follows the generated text.
    """
    lines = stdout.splitlines()
    # Find the start: line after the last "Throughput:" stats line
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Throughput:"):
            start = i + 1
    # Find the end: first '[' line after start
    end = len(lines)
    for i in range(start, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("["):
            end = i
            break
    text = "\n".join(lines[start:end]).strip()
    return text


def parse_perf_from_base_stdout(stdout: str) -> dict:
    """Extract performance metrics from base C++ binary (modeling_qwen3_omni) stdout.

    The base binary prints lines like:
        Prompt token size: 726
        Output token size: 53
        Preprocess time: 123.45 ms
        Vision encode time: 456.78 ms
        TTFT: 789.01 ms
        Decode time: 2345.67 ms
        TPOT: 45.07 ms/token
        Throughput: 22.19 tokens/s
    """
    perf = {}
    kv = parse_kv_stdout(stdout)
    def _float(key: str) -> float:
        val = kv.get(key, "0")
        # strip units like " ms", " ms/token", " tokens/s"
        for unit in (" ms/token", " tokens/s", " ms"):
            if val.endswith(unit):
                val = val[:-len(unit)]
                break
        try:
            return float(val)
        except ValueError:
            return 0.0

    perf["prompt_tokens"] = int(_float("Prompt token size"))
    perf["output_tokens"] = int(_float("Output token size"))
    perf["vision_encode_ms"] = _float("Vision encode time")
    perf["preprocess_ms"] = _float("Preprocess time")
    perf["ttft_ms"] = _float("TTFT")
    perf["decode_ms"] = _float("Decode time")
    perf["tpot_ms"] = _float("TPOT")
    perf["throughput"] = _float("Throughput")
    # Base binary doesn't output total_ms; compute from available stages
    perf["total_ms"] = perf["preprocess_ms"] + perf["vision_encode_ms"] + perf["ttft_ms"] + perf["decode_ms"]
    return perf


def run_cpp_case(cpp_bin: Path, model_dir: Path, image_path: Path, prompt: str, max_new_tokens: int, precision: str = "fp32", device: str = "CPU", timeout: int = 0) -> dict:
    cmd = [
        str(cpp_bin),
        "--model-dir", str(model_dir),
        "--image", str(image_path),
        "--prompt", prompt,
        "--device", device,
        "--output-tokens", str(max_new_tokens),
        "--precision", precision,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout if timeout > 0 else None)
    except subprocess.TimeoutExpired:
        return {
            "supported": True,
            "return_code": -999,
            "stdout": "",
            "stderr": f"TIMEOUT after {timeout}s",
            "text_output": "",
            "perf": {},
            "timeout": True,
        }
    perf = parse_perf_from_base_stdout(proc.stdout) if proc.returncode == 0 else {}
    result = {
        "supported": True,
        "return_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "text_output": parse_cpp_text(proc.stdout) if proc.returncode == 0 else "",
        "perf": perf,
    }
    return result


def parse_kv_stdout(stdout: str) -> dict:
    parsed = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def run_cpp_tts_case(
    cpp_tts_bin: Path,
    model_dir: Path,
    case_id: int,
    prompt: str,
    wav_out: Path,
    max_new_tokens: int,
    image_path: Path = None,
    audio_path: Path = None,
    device: str = "CPU",
    precision: str = "fp32",
    timeout: int = 0,
    video_frames_dir: Path = None,
) -> dict:
    cmd = [
        str(cpp_tts_bin),
        str(model_dir),
        str(case_id),
        prompt,
        str(wav_out),
        str(image_path) if image_path else "none",
        str(audio_path) if audio_path else "none",
        device,
        str(max_new_tokens),
        precision,
        str(video_frames_dir) if video_frames_dir else "none",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout if timeout > 0 else None)
    except subprocess.TimeoutExpired:
        return {
            "supported": True,
            "return_code": -999,
            "stdout": "",
            "stderr": f"TIMEOUT after {timeout}s",
            "text_output": "",
            "wav_output": "",
            "audio_samples": 0,
            "audio_sample_rate": 0,
            "tts_backend": "",
            "tts_note": "",
            "perf": {},
            "timeout": True,
        }
    parsed = parse_kv_stdout(proc.stdout) if proc.returncode == 0 else {}
    def _pf(key: str) -> float:
        try:
            return float(parsed.get(key, "0"))
        except ValueError:
            return 0.0

    perf = {}
    if proc.returncode == 0:
        perf = {
            "prompt_tokens": int(_pf("PROMPT_TOKENS")),
            "output_tokens": int(_pf("OUTPUT_TOKENS")),
            "model_load_ms": _pf("MODEL_LOAD_MS"),
            "audio_encode_ms": _pf("AUDIO_ENCODE_MS"),
            "vision_encode_ms": _pf("VISION_ENCODE_MS"),
            "ttft_ms": _pf("TTFT_MS"),
            "decode_ms": _pf("DECODE_MS"),
            "tpot_ms": _pf("TPOT_MS"),
            "throughput": _pf("THROUGHPUT"),
            "text_gen_ms": _pf("TEXT_GEN_MS"),
            "tts_ms": _pf("TTS_MS"),
            # TTS sub-component timing
            "tts_model_compile_ms": _pf("TTS_MODEL_COMPILE_MS"),
            "tts_talker_prefill_ms": _pf("TTS_TALKER_PREFILL_MS"),
            "tts_talker_decode_ms": _pf("TTS_TALKER_DECODE_MS"),
            "tts_code_predictor_ms": _pf("TTS_CODE_PREDICTOR_MS"),
            "tts_speech_decoder_ms": _pf("TTS_SPEECH_DECODER_MS"),
            "tts_codec_frames": int(_pf("TTS_CODEC_FRAMES")),
            "total_ms": _pf("TOTAL_MS"),
        }
    return {
        "supported": True,
        "return_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "text_output": parsed.get("TEXT_OUTPUT", "").replace("\\n", "\n") if proc.returncode == 0 else "",
        "wav_output": parsed.get("WAV_OUTPUT", "") if proc.returncode == 0 else "",
        "audio_samples": int(parsed.get("AUDIO_SAMPLES", "0")) if proc.returncode == 0 else 0,
        "audio_sample_rate": int(parsed.get("AUDIO_SAMPLE_RATE", "0")) if proc.returncode == 0 else 0,
        "tts_backend": parsed.get("TTS_BACKEND", "") if proc.returncode == 0 else "",
        "tts_note": parsed.get("TTS_NOTE", "") if proc.returncode == 0 else "",
        "perf": perf,
    }


def decode_python_generation(result, tokenizer, input_length: int):
    audio_keys = ["audio", "audios", "waveform", "waveforms", "tts", "speech"]
    text = ""
    has_audio = False

    if hasattr(result, "sequences"):
        sequences = result.sequences
        decoded = tokenizer.batch_decode(sequences[:, input_length:], skip_special_tokens=True)
        text = decoded[0] if decoded else ""
        for key in audio_keys:
            if hasattr(result, key):
                value = getattr(result, key)
                has_audio = value is not None
                break
        return text, has_audio

    if isinstance(result, dict):
        sequences = result.get("sequences")
        if sequences is not None:
            decoded = tokenizer.batch_decode(sequences[:, input_length:], skip_special_tokens=True)
            text = decoded[0] if decoded else ""
        for key in audio_keys:
            if key in result and result[key] is not None:
                has_audio = True
                break
        return text, has_audio

    if hasattr(result, "shape"):
        decoded = tokenizer.batch_decode(result[:, input_length:], skip_special_tokens=True)
        text = decoded[0] if decoded else ""
        return text, False

    return str(result), False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--cpp-bin", required=True)
    parser.add_argument("--cpp-tts-bin", required=False, default="")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--cpp-precision", type=str, default="",
                        help="(Deprecated) Single precision mode. Use --precisions instead.")
    parser.add_argument("--precisions", type=str, default="",
                        help="Comma-separated precision modes (e.g. 'fp32,fp16_kv8,inf_fp32_kv_fp32_w_int4_asym'). Default: fp32")
    parser.add_argument("--test-audio", type=str, default="",
                        help="Path to a real speech WAV file for case 3 (default: auto-generated 440Hz test tone)")
    parser.add_argument("--video", type=str, default="",
                        help="Path to video file for case 5 (e.g. rainning.mp4)")
    parser.add_argument("--case5-audio", type=str, default="",
                        help="Path to audio file for case 5 (e.g. thunder-and-rain-sounds.mp3). Falls back to --test-audio if not set.")
    parser.add_argument("--case5-image", type=str, default="",
                        help="Path to image file for case 5 (e.g. london.jpg). Falls back to --image if not set.")
    parser.add_argument("--case5-prompt-file", type=str, default="",
                        help="Path to text file containing the Case 5 prompt. If not set, uses a built-in default.")
    parser.add_argument("--max-video-frames", type=int, default=8,
                        help="Max video frames to extract for C++ binary (default: 8). "
                             "Higher values use more memory; 30 frames at 720p needs ~76GB for attention on CPU fp32.")
    parser.add_argument("--cpp-only", action="store_true",
                        help="Skip Python model loading and inference, only run C++ cases")
    parser.add_argument("--cases", type=str, default="",
                        help="Comma-separated list of case IDs to run (e.g. '4' or '3,4'). Empty = all.")
    parser.add_argument("--device", type=str, default="",
                        help="(Deprecated) Single device. Use --devices instead.")
    parser.add_argument("--devices", type=str, default="",
                        help="Comma-separated devices (e.g. 'CPU,GPU'). Default: CPU")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Timeout in seconds per C++ subprocess. 0=no timeout. Default: 600")
    args = parser.parse_args()

    # Resolve devices list (--devices takes precedence over deprecated --device)
    if args.devices:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    elif args.device:
        devices = [args.device.strip()]
    else:
        devices = ["CPU"]

    # Resolve precisions list (--precisions takes precedence over deprecated --cpp-precision)
    if args.precisions:
        precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    elif args.cpp_precision:
        precisions = [args.cpp_precision.strip()]
    else:
        precisions = ["fp32"]

    model_dir = Path(args.model_dir)
    image_path = Path(args.image)
    cpp_bin = Path(args.cpp_bin)
    cpp_tts_bin = Path(args.cpp_tts_bin) if args.cpp_tts_bin else None
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    test_audio = out_json.parent / "qwen3_omni_test_tone.wav"
    ensure_test_wav(test_audio)
    if args.test_audio:
        test_audio = Path(args.test_audio)
        if not test_audio.exists():
            raise FileNotFoundError(f"Test audio not found: {test_audio}")

    # --- Case 5 resources ---
    case5_image = Path(args.case5_image) if args.case5_image else image_path
    case5_audio = Path(args.case5_audio) if args.case5_audio else test_audio
    case5_video = Path(args.video) if args.video else None
    case5_prompt = (
        "You are a weather bot. I'm showing you my current location and a forecast report. "
        "Look at the window (video) and listen to the environment. "
        "Is the forecast accurate? Respond with a summary and a voice alert."
    )
    if args.case5_prompt_file:
        pf = Path(args.case5_prompt_file)
        if pf.exists():
            case5_prompt = pf.read_text(encoding="utf-8").strip()

    case_filter = set(int(x) for x in args.cases.split(",") if x.strip()) if args.cases else set()

    model = None
    processor = None
    process_mm_info = None
    if not args.cpp_only:
        base = Path(__file__).resolve().parents[2] / "qwen3_omni_4B_final_release"
        internal = glob.glob(str(base / "transformers-internal-q3o-dense-*" / "src"))
        if not internal:
            raise RuntimeError("Cannot locate internal transformers source")
        sys.path.insert(0, internal[0])
        sys.path.insert(0, str(base))

        from transformers import Qwen3OmniForConditionalGeneration, Qwen3OmniProcessor
        from qwen_omni_utils import process_mm_info as _process_mm_info
        process_mm_info = _process_mm_info

        model = Qwen3OmniForConditionalGeneration.from_pretrained(
            str(model_dir),
            dtype="auto",
            device_map="cpu",
            trust_remote_code=True,
        )
        model.eval()
        processor = Qwen3OmniProcessor.from_pretrained(str(model_dir), trust_remote_code=True, use_fast=False)

    # Prompt from Python demo (qwen3_omni_4b_demo/run_qwen3_omni_dense.py)
    demo_prompt = "What can you see in the image and hear in the audio wav? Answer in one short sentence."

    cases = [
        {
            "id": 1,
            "name": "image+text -> text",
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": "Describe this image in detail."},
                    ],
                }
            ],
            "cpp_supported": True,
            "expect_tts": False,
        },
        {
            "id": 2,
            "name": "image+text -> text+tts",
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": "Describe this image and also output speech audio."},
                    ],
                }
            ],
            "cpp_supported": cpp_tts_bin is not None,
            "cpp_mode": "tts_min",
            "cpp_prompt": "Describe this image and provide a speech response.",
            "expect_tts": True,
        },
        {
            "id": 3,
            "name": "audio+text -> text+tts",
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": str(test_audio)},
                        {"type": "text", "text": "What sound do you hear in the audio? Answer in one short sentence."},
                    ],
                }
            ],
            "cpp_supported": cpp_tts_bin is not None,
            "cpp_mode": "tts_min",
            "cpp_prompt": "What sound do you hear in the audio? Answer in one short sentence.",
            "expect_tts": True,
        },
        {
            "id": 4,
            "name": "image+audio+text -> text+tts (demo)",
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "audio", "audio": str(test_audio)},
                        {"type": "text", "text": demo_prompt},
                    ],
                }
            ],
            "cpp_supported": cpp_tts_bin is not None,
            "cpp_mode": "tts_min",
            "cpp_prompt": demo_prompt,
            "expect_tts": True,
        },
    ]

    # --- Case 5: text + image + video + audio -> text + tts ---
    # Auto-extract video frames for C++ binary when video is available
    case5_video_frames_dir = None
    if case5_video and case5_video.exists() and cpp_tts_bin is not None:
        case5_video_frames_dir = out_json.parent / "case5_video_frames"

        # Prefer C++ extract_video_frames binary (no Python deps needed)
        cpp_bin_dir = Path(cpp_tts_bin).resolve().parent if cpp_tts_bin else None
        extract_cpp_bin = None
        if cpp_bin_dir:
            for name in ("extract_video_frames.exe", "extract_video_frames"):
                candidate = cpp_bin_dir / name
                if candidate.exists():
                    extract_cpp_bin = candidate
                    break

        if extract_cpp_bin:
            print(f"[Case5] Extracting video frames from {case5_video} (C++ backend) ...")
            extract_cmd = [
                str(extract_cpp_bin),
                "--video", str(case5_video),
                "--output-dir", str(case5_video_frames_dir),
            ]
            if args.max_video_frames:
                extract_cmd += ["--max-frames", str(args.max_video_frames)]
            rc = subprocess.run(extract_cmd, capture_output=True, text=True)
            if rc.returncode != 0:
                print(f"[Case5] WARNING: C++ frame extraction failed: {rc.stderr}")
                case5_video_frames_dir = None
            else:
                print(f"[Case5] Frames saved to {case5_video_frames_dir}")
        else:
            # Fallback to Python script
            extract_script = Path(__file__).resolve().parent / "extract_video_frames.py"
            if extract_script.exists():
                print(f"[Case5] Extracting video frames from {case5_video} (Python fallback) ...")
                extract_cmd = [
                    sys.executable, str(extract_script),
                    "--video", str(case5_video),
                    "--output-dir", str(case5_video_frames_dir),
                ]
                if args.max_video_frames:
                    extract_cmd += ["--max-frames", str(args.max_video_frames)]
                rc = subprocess.run(extract_cmd, capture_output=True, text=True)
                if rc.returncode != 0:
                    print(f"[Case5] WARNING: frame extraction failed: {rc.stderr}")
                    case5_video_frames_dir = None
                else:
                    print(f"[Case5] Frames saved to {case5_video_frames_dir}")
            else:
                print(f"[Case5] WARNING: no frame extraction tool found (neither C++ nor Python)")
                case5_video_frames_dir = None

    case5_content = [{"type": "image", "image": str(case5_image)}]
    if case5_video and case5_video.exists():
        case5_content.append({"type": "video", "video": str(case5_video)})
    case5_content.append({"type": "audio", "audio": str(case5_audio)})
    case5_content.append({"type": "text", "text": case5_prompt})
    cases.append({
        "id": 5,
        "name": "image+video+audio+text -> text+tts",
        "conversation": [{"role": "user", "content": case5_content}],
        "cpp_supported": cpp_tts_bin is not None,
        "cpp_mode": "tts_min",
        "cpp_prompt": case5_prompt,
        "expect_tts": True,
        "use_audio_in_video": False,
        "cpp_image": case5_image,   # may differ from the global --image
        "cpp_audio": case5_audio,   # may differ from --test-audio
        "cpp_video_frames_dir": case5_video_frames_dir,
    })

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(model_dir),
        "image": str(image_path),
        "audio": str(test_audio),
        "case5_image": str(case5_image),
        "case5_video": str(case5_video) if case5_video else "",
        "case5_audio": str(case5_audio),
        "devices": devices,
        "precisions": precisions,
        "results": [],
    }

    total_combos = len(devices) * len(precisions) * (len([c for c in cases if not case_filter or c["id"] in case_filter]))
    combo_idx = 0

    for prec in precisions:
        for device in devices:
            for case in cases:
                if case_filter and case["id"] not in case_filter:
                    continue

                combo_idx += 1
                combo_label = f"[{combo_idx}/{total_combos}] precision={prec} device={device} case={case['id']}"
                print(f"\n>>> {combo_label} <<<")
                print(f"  [{case['id']}] {case['name']}")
                # Extract the text prompt from conversation
                conv_text = ""
                for msg in case["conversation"]:
                    content = msg.get("content")
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                conv_text = item.get("text", "")
                print(f"  Prompt: text=\"{conv_text}\"")
                if case.get("cpp_prompt"):
                    print(f"  C++ Prompt: cpp_prompt=\"{case['cpp_prompt']}\"")

                case_result = {
                    "id": case["id"],
                    "name": case["name"],
                    "device": device,
                    "precision": prec,
                    "expect_tts": case["expect_tts"],
                    "cpp": {"supported": case["cpp_supported"]},
                    "python": {},
                }

                if not args.cpp_only and processor is not None:
                    use_audio_in_video = case.get("use_audio_in_video", True)
                    text = processor.apply_chat_template(case["conversation"], add_generation_prompt=True, tokenize=False)
                    audios, images, videos = process_mm_info(case["conversation"], use_audio_in_video=use_audio_in_video)

                    try:
                        inputs = processor(
                            text=text,
                            audio=audios,
                            images=images,
                            videos=videos,
                            return_tensors="pt",
                            padding=True,
                            use_audio_in_video=use_audio_in_video,
                        )
                        inputs = inputs.to(model.device).to(model.dtype)
                        generation = model.generate(
                            **inputs,
                            do_sample=False,
                            max_new_tokens=args.max_new_tokens,
                            speaker="f245",
                            return_audio=case["expect_tts"],
                            use_audio_in_video=use_audio_in_video,
                            thinker_return_dict_in_generate=True,
                        )

                        py_has_audio = False
                        generation_for_decode = generation
                        if isinstance(generation, tuple):
                            generation_for_decode = generation[0]
                            py_has_audio = generation[1] is not None

                        py_text, py_has_audio_from_decode = decode_python_generation(
                            generation_for_decode,
                            processor.tokenizer,
                            int(inputs["input_ids"].shape[1]),
                        )
                        py_has_audio = py_has_audio or py_has_audio_from_decode
                        case_result["python"] = {
                            "ok": True,
                            "text_output": py_text,
                            "has_tts_audio": py_has_audio,
                        }
                    except Exception as error:
                        case_result["python"] = {
                            "ok": False,
                            "error": str(error),
                        }
                else:
                    case_result["python"] = {"ok": False, "error": "skipped (--cpp-only)"}

                if case["cpp_supported"]:
                    if case.get("cpp_mode") == "tts_min":
                        wav_out = out_json.parent / f"case{case['id']}_{device}_{prec}_cpp_tts.wav"
                        # Only pass image_path / audio_path when the case conversation actually uses them
                        case_has_image = any(
                            c.get("type") == "image"
                            for msg in case["conversation"]
                            for c in (msg.get("content") if isinstance(msg.get("content"), list) else [])
                        )
                        case_has_audio = any(
                            c.get("type") == "audio"
                            for msg in case["conversation"]
                            for c in (msg.get("content") if isinstance(msg.get("content"), list) else [])
                        )
                        # Case 5 may use its own image / audio files
                        eff_image = case.get("cpp_image", image_path) if case_has_image else None
                        eff_audio = case.get("cpp_audio", test_audio) if case_has_audio else None
                        eff_video = case.get("cpp_video_frames_dir")
                        case_result["cpp"] = run_cpp_tts_case(
                            cpp_tts_bin,
                            model_dir,
                            case["id"],
                            case.get("cpp_prompt", ""),
                            wav_out,
                            args.max_new_tokens,
                            image_path=eff_image,
                            audio_path=eff_audio,
                            device=device,
                            precision=prec,
                            timeout=args.timeout,
                            video_frames_dir=eff_video,
                        )
                    else:
                        cpp_prompt = "Describe this image in detail."
                        case_result["cpp"] = run_cpp_case(cpp_bin, model_dir, image_path, cpp_prompt, args.max_new_tokens, prec, device=device, timeout=args.timeout)
                else:
                    case_result["cpp"] = {
                        "supported": False,
                        "reason": "C++ tts sample binary is not provided. Pass --cpp-tts-bin to enable case2/3/4.",
                    }

                # Print quick status line
                cpp = case_result["cpp"]
                if cpp.get("timeout"):
                    print(f"  C++: TIMEOUT (>{args.timeout}s)  --  skipped")
                elif cpp.get("return_code") == 0:
                    print(f"  C++: OK")
                    cpp_text = cpp.get("text_output", "")
                    if len(cpp_text) > 200:
                        cpp_text = cpp_text[:200] + "..."
                    if cpp_text:
                        print(f"  C++ text: {cpp_text}")
                    if cpp.get("wav_output"):
                        print(f"  C++ wav: {cpp['wav_output']}")
                    if cpp.get("audio_samples"):
                        print(f"  C++ audio_samples: {cpp['audio_samples']}")
                    if cpp.get("tts_backend"):
                        print(f"  C++ tts_backend: {cpp['tts_backend']}")
                    # Performance metrics
                    perf = cpp.get("perf", {})
                    if perf:
                        parts = []
                        if perf.get("output_tokens"):
                            parts.append(f"tokens={perf['output_tokens']}")
                        if perf.get("ttft_ms"):
                            parts.append(f"TTFT={perf['ttft_ms']:.0f}ms")
                        if perf.get("tpot_ms"):
                            parts.append(f"TPOT={perf['tpot_ms']:.1f}ms")
                        if perf.get("throughput"):
                            parts.append(f"throughput={perf['throughput']:.1f}t/s")
                        if perf.get("total_ms"):
                            parts.append(f"total={perf['total_ms']:.0f}ms")
                        elif perf.get("text_gen_ms"):
                            parts.append(f"total={perf.get('text_gen_ms',0)+perf.get('tts_ms',0):.0f}ms")
                        if parts:
                            print(f"  Perf: {', '.join(parts)}")
                elif cpp.get("supported"):
                    stderr_text = cpp.get('stderr', '')
                    if 'Exceeded max size of memory' in stderr_text or 'out of memory' in stderr_text.lower():
                        print(f"  C++: OOM (GPU memory allocation exceeded)")
                        cpp['oom'] = True
                    else:
                        print(f"  C++: FAIL (rc={cpp.get('return_code')})")
                    if stderr_text:
                        # Show first meaningful line of stderr
                        for line in stderr_text.splitlines():
                            if line.strip() and ('Error' in line or 'Exceeded' in line or 'failed' in line.lower()):
                                print(f"  C++ stderr: {line.strip()[:150]}")
                                break

                report["results"].append(case_result)

        # --- Per-precision summary after all devicesxcases for this precision ---
        prec_rows = [(r["device"], r["id"], r["name"], r["cpp"].get("perf", {}))
                     for r in report["results"]
                     if r["precision"] == prec
                     and r["cpp"].get("supported") and r["cpp"].get("return_code") == 0
                     and r["cpp"].get("perf")]
        if prec_rows:
            print(f"\n{'='*110}")
            print(f" Precision Summary: {prec}")
            print(f"{'='*110}")
            header = f"{'Device':<6} {'Case':<35} {'Tokens':>7} {'TTFT':>9} {'Decode':>9} {'TPOT':>9} {'Thru':>9} {'Total':>10}"
            print(header)
            print("-" * len(header))
            for dev, cid, cname, perf in prec_rows:
                tokens = f"{perf.get('output_tokens', 0)}"
                ttft = f"{perf['ttft_ms']:.0f}ms" if perf.get("ttft_ms") else "N/A"
                decode = f"{perf['decode_ms']:.0f}ms" if perf.get("decode_ms") else "N/A"
                tpot = f"{perf['tpot_ms']:.1f}ms" if perf.get("tpot_ms") else "N/A"
                thru = f"{perf['throughput']:.1f}t/s" if perf.get("throughput") else "N/A"
                total = f"{perf['total_ms']:.0f}ms" if perf.get("total_ms") else (
                    f"{perf.get('text_gen_ms', 0) + perf.get('tts_ms', 0):.0f}ms" if perf.get("text_gen_ms") else "N/A"
                )
                name_short = cname[:32] if len(cname) > 32 else cname
                print(f"{dev:<6} [{cid}] {name_short:<32} {tokens:>7} {ttft:>9} {decode:>9} {tpot:>9} {thru:>9} {total:>10}")
            print(f"{'='*110}")

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ============================================================
    # Grand Summary  --  all precisions x devices x cases in one table
    # ============================================================
    all_perf = [(r["precision"], r["device"], r["id"], r["name"], r["cpp"].get("perf", {}))
                for r in report["results"]
                if r["cpp"].get("supported") and r["cpp"].get("return_code") == 0
                and r["cpp"].get("perf")]
    if all_perf:
        print(f"\n{'#'*130}")
        print(f"  GRAND PERFORMANCE SUMMARY  ({len(all_perf)} runs)")
        print(f"{'#'*130}")
        header = f"{'Precision':<32} {'Device':<6} {'Case':<35} {'Tokens':>7} {'TTFT':>9} {'Decode':>9} {'TPOT':>9} {'Thru':>9} {'Total':>10}"
        print(header)
        print("-" * len(header))
        prev_prec = None
        for prec, dev, cid, cname, perf in all_perf:
            if prev_prec is not None and prec != prev_prec:
                print()  # blank line between precision groups
            prev_prec = prec
            tokens = f"{perf.get('output_tokens', 0)}"
            ttft = f"{perf['ttft_ms']:.0f}ms" if perf.get("ttft_ms") else "N/A"
            decode = f"{perf['decode_ms']:.0f}ms" if perf.get("decode_ms") else "N/A"
            tpot = f"{perf['tpot_ms']:.1f}ms" if perf.get("tpot_ms") else "N/A"
            thru = f"{perf['throughput']:.1f}t/s" if perf.get("throughput") else "N/A"
            total = f"{perf['total_ms']:.0f}ms" if perf.get("total_ms") else (
                f"{perf.get('text_gen_ms', 0) + perf.get('tts_ms', 0):.0f}ms" if perf.get("text_gen_ms") else "N/A"
            )
            name_short = cname[:32] if len(cname) > 32 else cname
            print(f"{prec:<32} {dev:<6} [{cid}] {name_short:<32} {tokens:>7} {ttft:>9} {decode:>9} {tpot:>9} {thru:>9} {total:>10}")
        print(f"{'#'*130}")

    # Also list any failures / timeouts
    fails = [r for r in report["results"]
             if r["cpp"].get("supported") and r["cpp"].get("return_code", -1) != 0]
    if fails:
        oom_count = sum(1 for r in fails if r["cpp"].get("oom"))
        other_count = len(fails) - oom_count
        label_parts = []
        if oom_count: label_parts.append(f"{oom_count} OOM")
        if other_count: label_parts.append(f"{other_count} FAIL")
        timeout_count = sum(1 for r in fails if r["cpp"].get("timeout"))
        if timeout_count: label_parts.append(f"{timeout_count} TIMEOUT")
        print(f"\n  FAILED / TIMED-OUT RUNS ({len(fails)}: {', '.join(label_parts)}):")
        for r in fails:
            if r["cpp"].get("timeout"):
                reason = "TIMEOUT"
            elif r["cpp"].get("oom"):
                reason = "OOM  --  GPU memory allocation exceeded max single alloc size"
            else:
                reason = f"rc={r['cpp'].get('return_code')}"
            print(f"    precision={r['precision']}  device={r['device']}  case={r['id']}  {reason}")

    print(f"\nSaved report: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
