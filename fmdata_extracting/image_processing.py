import cv2
import os

data = []
for i in range(1, 12):
    data.append(f'home_{i}')
    data.append(f'away_{i}')

def image_capture(videos):
    for vi in videos:
        # 비디오 파일 경로
        video_path = f'./videos/{vi}.mp4'
        output_folder = f'./images/{vi}'

        # 출력 폴더 생성
        os.makedirs(output_folder, exist_ok=True)

        # 비디오 로드
        cap = cv2.VideoCapture(video_path)

        # FPS 및 총 프레임 수 확인
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # 원본 영상 FPS
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 전체 프레임 수
        duration = frame_count / fps  # 영상 길이 (초)

        print(f"Original FPS: {fps}, Total Frames: {frame_count}, Duration: {duration} sec")

        # 25fps로 프레임 추출
        target_fps = 25
        frame_interval = fps / target_fps

        frame_id = 0
        save_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % int(frame_interval) == 0:
                frame_filename = os.path.join(output_folder, f"frame_{save_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                save_count += 1

            frame_id += 1

        cap.release()
        print(f"Extracted {save_count} frames at {target_fps} FPS. from {vi}")
        

def image_cropping(images):
    for im in images:
        # 입력 폴더와 출력 폴더 경로
        input_folder = f"images/{im}"
        output_folder = f"cropped_images/{im}"

        # 출력 폴더 없으면 생성
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 자를 영역 정의 (y_start:y_end, x_start:x_end) - 원하는 영역으로 변경 가능
        y_start, y_end = 150, 604   # 높이 (세로)
        x_start, x_end = 914, 1620 # 너비 (가로)

        # 파일 리스트 가져오기
        image_files = sorted([f for f in os.listdir(input_folder) if f.startswith("frame_")])

        # 이미지 크롭 및 저장
        for i, filename in enumerate(image_files):
            image_path = os.path.join(input_folder, filename)
            output_filename = f"frame_{i:04d}.jpg"  # 새로운 이름
            output_path = os.path.join(output_folder, output_filename)

            # 이미지 불러오기
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading {filename}")
                continue

            # 이미지 크롭
            cropped = image[y_start:y_end, x_start:x_end]

            # 크롭된 이미지 저장
            cv2.imwrite(output_path, cropped)
            if i%100 == 0:
                print(f"Saved cropped image: {output_path}")

        print("✅ 모든 이미지 크롭 완료!")

#image_capture(data)
image_cropping(data)