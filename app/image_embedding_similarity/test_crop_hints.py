"""
Crop Hints 단독 테스트 스크립트.
Qdrant, FastAPI, 임베딩 전부 빼고 "크롭이 실제로 잘 되는지"만 확인한다.

사용법:
    uv run python test_crop_hints.py /path/to/image.jpg

결과:
    같은 폴더에 {원본이름}_cropped.jpg 파일이 생성됨 -> 직접 열어서 확인
    콘솔에 신뢰도/박스 좌표도 출력됨
"""
import sys
from pathlib import Path

from dotenv import load_dotenv

# .env 파일을 환경변수로 로드 (crop_utils가 GOOGLE_APPLICATION_CREDENTIALS를
# 읽으려면 vision.ImageAnnotatorClient()가 생성되기 전에 로드되어야 한다)
load_dotenv()

from PIL import Image

from crop_utils import crop_with_padding, get_crop_hint_box


def main():
    if len(sys.argv) != 2:
        print("사용법: uv run python test_crop_hints.py /path/to/image.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"파일을 찾을 수 없습니다: {image_path}")
        sys.exit(1)

    image_bytes = image_path.read_bytes()
    original_image = Image.open(image_path).convert("RGB")
    print(f"원본 크기: {original_image.size}")

    box = get_crop_hint_box(image_bytes, original_image.size)

    if not box:
        print("\n조건을 만족하는 크롭 영역이 없어 원본을 그대로 사용합니다.")
        return

    cropped = crop_with_padding(original_image, box)
    out_path = image_path.with_name(image_path.stem + "_cropped.jpg")
    cropped.save(out_path)
    print(f"\n크롭된 이미지 저장 완료: {out_path}")
    print(f"크롭 후 크기: {cropped.size}")


if __name__ == "__main__":
    main()