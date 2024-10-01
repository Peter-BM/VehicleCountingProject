import streamlit as st
import cv2
from ultralytics import YOLO, solutions
import tempfile


def main():
    st.title("Gerenciamento de Volume de Tráfego")

    # Load the OpenVINO model
    ov_model = YOLO("yolov8s_openvino_model/")

    with st.sidebar:
        st.title('Configurações de detecção de objetos')
        with st.form("Parameters: "):
            classes = ['Car', 'Motorcycle', 'Bus', 'Truck', 'Person']
            class_indices = {'Car': 2, 'Motorcycle': 3, 'Bus': 5, 'Truck': 7, 'Person': 0}
            selected_classes = st.multiselect('Objetos a detectar:', classes, default=classes)
            classes_to_count = [class_indices[cls] for cls in selected_classes]

            st.divider()

            line_y = st.slider("Posição linha no eixo y", min_value=0, max_value=720, value=550)
            left_x = st.slider("Posição do ponto 0X", min_value=0, max_value=1600, value=100)
            rigth_x = st.slider("Posição do ponto 1X", min_value=0, max_value=1600, value=1300)

            line_points = [(left_x, line_y), (rigth_x, line_y)]

            st.divider()

            uploaded_file = st.file_uploader("Escolha um vídeo...", type=["mp4", "avi"])
            tfile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            submit = st.form_submit_button("Submeter", type="primary")

    DEMO_VIDEO = '/home/meneghini_/Desktop/faculdade/TCC/TCC1/imagensEVideos/RuaDia.mp4'

    if submit:
        if uploaded_file is not None:
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        else:
            video_path = DEMO_VIDEO

        # Exibe o vídeo original
        dem_vid = open(video_path, 'rb')
        demo_bytes = dem_vid.read()
        st.sidebar.text('Vídeo Original')
        st.sidebar.video(demo_bytes)

        # Abre o vídeo
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Inicializa o contador
        counter = solutions.ObjectCounter(
            view_img=False,
            reg_pts=line_points,
            names=ov_model.names,
            draw_tracks=False,
            line_thickness=2,
        )

        # Processa os frames do vídeo
        frame_number = 0
        progress_bar = st.progress(0)

        with st.spinner('Processando o vídeo...'):
            placeholder = st.empty()  # Cria um placeholder fazio
            while cap.isOpened():
                success, im0 = cap.read()
                if not success:
                    print("Video frame is empty or video processing has been successfully completed.")
                    break
                tracks = ov_model.track(im0, persist=True, show=False, classes=classes_to_count)
                im0 = counter.start_counting(im0, tracks)

                placeholder.image(im0, channels="BGR")  # Update the placeholder with the new frame

                frame_number += 1
                if total_frames > 0:
                    progress_bar.progress(frame_number / total_frames)
                else:
                    progress_bar.progress(1)

        cap.release()

        cv2.destroyAllWindows()

        # Display counts and other info
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("***FPS:***")
            st.markdown(fps)
        with col2:
            st.markdown("***Objetos contados:***")
            counts_text = ""
            total_count = 0
            for cls_idx in classes_to_count:
                cls_name = ov_model.names[cls_idx]
                cls_count = counter.in_counts + counter.out_counts
                counts_text += f"{cls_name}: {cls_count}<br>"
                total_count += cls_count
            st.markdown(counts_text, unsafe_allow_html=True)
            st.markdown(f'Total count: {total_count}')
        with col3:
            st.markdown("***Largura:***")
            st.markdown(w)

    else:
        st.write("Por favor, ajuste as configurações e pressione 'Submeter' para iniciar a detecção.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass