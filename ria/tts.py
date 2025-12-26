"""
TTS 모듈 - OpenAI TTS API를 이용한 음성 합성
"""

from pathlib import Path

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI
from scipy import signal

# .env 파일 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# OpenAI 클라이언트 생성
client = OpenAI()

# 오디오 출력 설정
OUTPUT_SAMPLE_RATE = 24000  # OpenAI TTS 기본 샘플레이트
OUTPUT_CHANNELS = 1
OUTPUT_DEVICE_NAME = "Voicemeeter Point 1"  # Voicemeeter Point 1 출력


def synthesize(text: str, voice: str = "nova") -> bytes:
    """
    텍스트를 PCM 음성 데이터로 변환합니다.

    Args:
        text: 변환할 텍스트
        voice: 음성 종류 (alloy, echo, fable, onyx, nova, shimmer)

    Returns:
        PCM 오디오 데이터 (bytes)
    """
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="pcm",  # PCM 형식으로 받기
    )
    return response.content


def _find_device_index(name_contains: str) -> int | None:
    """장치 이름에 특정 문자열이 포함된 출력 장치의 인덱스를 찾습니다."""
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] > 0 and name_contains in dev["name"]:
            return i
    return None


def play(text: str, voice: str = "nova", device_name: str = OUTPUT_DEVICE_NAME):
    """
    텍스트를 음성으로 변환하여 바로 재생합니다.

    Args:
        text: 변환할 텍스트
        voice: 음성 종류 (alloy, echo, fable, onyx, nova, shimmer)
        device_name: 출력 장치 이름 (포함된 문자열로 검색)
    """
    # 장치 찾기
    device_index = _find_device_index(device_name)
    if device_index is None:
        raise RuntimeError(f"출력 장치를 찾을 수 없습니다: {device_name}")

    device_info = sd.query_devices(device_index)
    device_sample_rate = int(device_info["default_samplerate"])

    # TTS 스트리밍으로 오디오 데이터 수집
    audio_buffer = b""
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="pcm",
    ) as response:
        for chunk in response.iter_bytes(chunk_size=4096):
            audio_buffer += chunk

    # PCM 데이터를 numpy 배열로 변환
    audio = np.frombuffer(audio_buffer, dtype=np.int16)

    # 24000Hz -> 장치 샘플레이트로 리샘플링 (필요시)
    if device_sample_rate != OUTPUT_SAMPLE_RATE:
        num_samples = int(len(audio) * device_sample_rate / OUTPUT_SAMPLE_RATE)
        audio = signal.resample(audio, num_samples).astype(np.int16)

    # int16 -> float32 변환 (sounddevice는 float32 사용)
    audio_float = audio.astype(np.float32) / 32768.0

    # 재생
    sd.play(audio_float, samplerate=device_sample_rate, device=device_index)
    sd.wait()  # 재생 완료 대기


def synthesize_to_file(text: str, output_path: str, voice: str = "nova") -> str:
    """
    텍스트를 음성으로 변환하여 파일로 저장합니다.

    Args:
        text: 변환할 텍스트
        output_path: 출력 파일 경로
        voice: 음성 종류 (alloy, echo, fable, onyx, nova, shimmer)

    Returns:
        저장된 파일 경로
    """
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=text,
    ) as response:
        response.stream_to_file(output_path)
    return output_path


def list_output_devices():
    """사용 가능한 출력 장치 목록을 출력합니다."""
    print("사용 가능한 출력 장치:")
    print("-" * 60)
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] > 0:
            name = dev["name"]
            channels = dev["max_output_channels"]
            rate = int(dev["default_samplerate"])
            print(f"  [{i:2d}] {name} (ch:{channels}, rate:{rate})")
    print("-" * 60)
    print("TIP: VoiceMeeter로 보내려면 'Voicemeeter Point N' 장치 사용")


if __name__ == "__main__":
    # 테스트
    print("TTS 테스트")
    print("-" * 40)

    # 출력 장치 목록 표시
    list_output_devices()
    print()

    test_text = "안녕하세요! 저는 리아입니다. 만나서 반갑습니다."
    print(f"텍스트: {test_text}")
    print("[재생 중...]")

    play(test_text)
    print("[재생 완료]")
