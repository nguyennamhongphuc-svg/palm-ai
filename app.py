import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="AI Xem Chỉ Tay Pro", page_icon="🔮", layout="centered")

st.title("🔮 AI ĐỌC CHỈ TAY XOAY GÓC TỰ ĐỘNG")
st.caption("Phiên bản cấp cao: Tự động xoay góc và khớp đường cong theo mọi dáng tay.")

# ==========================================
# THUẬT TOÁN VẼ ĐƯỜNG CONG XOAY THEO KHUNG XƯƠNG
# ==========================================
def draw_adaptive_bezier(img, p0, p1, p2, color, thickness=5):
    """Vẽ đường cong luôn bám sát theo các khớp ngón tay"""
    t_values = np.linspace(0, 1, 30)
    points = []
    for t in t_values:
        x = int((1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0])
        y = int((1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1])
        points.append((x, y))
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], color, thickness)

# ==========================================
# BƯỚC 2: PHÂN TÍCH MA TRẬN HÌNH HỌC TỪ OPENCV
# ==========================================
def extract_line_features(img_raw):
    img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    edges = cv2.Canny(img_blur, 30, 90)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    features = {}
    line_keys = ['Heart_Line', 'Head_Line', 'Life_Line']

    for i, key in enumerate(line_keys):
        if i < len(sorted_contours):
            cnt = sorted_contours[i]
            length = cv2.arcLength(cnt, False)
            clarity = cv2.Laplacian(img_gray, cv2.CV_64F).var()

            if len(cnt) >= 2:
                p_start, p_end = cnt[0][0], cnt[-1][0]
                euclidean_dist = np.sqrt((p_start[0] - p_end[0])**2 + (p_start[1] - p_end[1])**2)
                curvature = length / (euclidean_dist + 1e-5)
            else: curvature = 1.0
            features[key] = {'length': length, 'curvature': curvature, 'clarity': clarity}
        else:
            features[key] = {'length': 0.0, 'curvature': 1.0, 'clarity': 0.0}
    return features

def run_rule_engine(features):
    results = {}
    life_len = features['Life_Line']['length']
    life_clar = features['Life_Line']['clarity']
    life_curv = features['Life_Line']['curvature']
    if life_len > 380: results['Life_Line'] = 'Super_Long_Vigorous' if life_clar > 70 else 'Long_Branch_Out'
    elif life_len > 220: results['Life_Line'] = 'Medium_Curved_Active' if life_curv > 1.35 else 'Medium_Straight_Stable'
    else: results['Life_Line'] = 'Short_Deep_Dense' if life_clar > 80 else 'Short_Fragile'

    heart_curv = features['Heart_Line']['curvature']
    heart_len = features['Heart_Line']['length']
    if heart_curv > 1.55: results['Heart_Line'] = 'Extreme_Curved_Passionate'
    elif heart_curv > 1.15: results['Heart_Line'] = 'Straight_Long_Altruist' if heart_len > 280 else 'Straight_Short_Cold'
    else: results['Heart_Line'] = 'Chain_Broken_Anxious'

    head_clar = features['Head_Line']['clarity']
    head_len = features['Head_Line']['length']
    head_curv = features['Head_Line']['curvature']
    if head_len > 320: results['Head_Line'] = 'Long_Curved_Creative' if head_curv > 1.25 else 'Long_Straight_Mastermind'
    elif head_len > 180: results['Head_Line'] = 'Medium_Balanced_Adaptable'
    else: results['Head_Line'] = 'Short_Material_Fast' if head_clar > 75 else 'Short_Faint_Distracted'
    return results

# ==========================================
# BƯỚC 3: GIAO DIỆN CAMERA CHÍNH
# ==========================================
camera_image = st.camera_input("Xòe tay trước camera (bạn có thể xoay nghiêng tùy ý):")

if camera_image is not None:
    file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_draw = img_bgr.copy()
    h, w, _ = img_bgr.shape

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, model_complexity=1) as hands:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            st.success("✅ Đã khóa mục tiêu bàn tay! Đường cong tự động xoay chuyển theo góc nghiêng.")
            landmarks = results.multi_hand_landmarks[0].landmark
            
            def get_pt(idx): return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
            
            # Lấy các điểm khớp xương thực tế chống xoay lệch
            wrist = get_pt(0)
            thumb_cmc = get_pt(1)
            thumb_mcp = get_pt(2)
            index_mcp = get_pt(5)
            middle_mcp = get_pt(9)
            ring_mcp = get_pt(13)
            pinky_mcp = get_pt(17)

            # Gốc khởi tạo các đường chỉ tay thực tế
            palm_edge_center = (int((wrist[0] + pinky_mcp[0])/2), int((wrist[1] + pinky_mcp[1])/2))
            between_index_thumb = (int((index_mcp[0] + thumb_mcp[0])/2), int((index_mcp[1] + thumb_mcp[1])/2))

            # 1. Vẽ ĐƯỜNG TÂM ĐẠO (Đỏ) - Đi từ gốc ngón út bám qua ngón áp út sang ngón trỏ
            draw_adaptive_bezier(img_draw, pinky_mcp, ring_mcp, index_mcp, (0, 0, 255), 5)

            # 2. Vẽ ĐƯỜNG TRÍ ĐẠO (Xanh dương) - Đi từ kẽ tay qua lòng bàn tay hướng về cạnh tay
            draw_adaptive_bezier(img_draw, between_index_thumb, middle_mcp, palm_edge_center, (255, 0, 0), 5)

            # 3. Vẽ ĐƯỜNG SINH ĐẠO (Xanh lá) - Đi từ kẽ tay ôm trọn gò ngón cái xuống cổ tay
            draw_adaptive_bezier(img_draw, between_index_thumb, thumb_cmc, wrist, (0, 255, 0), 5)

            # Hiển thị ảnh kết quả lên giao diện
            st.image(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB), caption="AI phân tích cấu trúc chỉ tay thích ứng hình học", use_container_width=True)

            # Đọc luận giải luận tướng
            features = extract_line_features(img_bgr)
            classified = run_rule_engine(features)

            st.markdown("## 📊 KẾT QUẢ DỰ ĐOÁN HÌNH DÁNG CHỈ TAY")
            
            # Sinh đạo
            st.subheader("1. 🟢 ĐƯỜNG SINH ĐẠO (Vận Mệnh & Sức Khỏe)")
            life = classified.get('Life_Line')
            if life == 'Super_Long_Vigorous':
                st.markdown("**Hình học:** Đường rất dài, nét sâu đậm rõ ràng.\n\n**Hướng đi:** Chạy thành một đường vòng cung lớn, ôm trọn lấy gò Kim Tinh (vùng thịt ngón cái) kéo dài sát cổ tay.\n\n**Luận tướng:** Thể lực dồi dào, tràn đầy sinh khí, có sức đề kháng bẩm sinh rất tốt.")
            elif life == 'Long_Branch_Out':
                st.markdown("**Hình học:** Đường dài, phần đuôi xuất hiện nhánh chẻ.\n\n**Hướng đi:** Đường cong chạy xuống hướng cổ tay nhưng có xu hướng rẽ nhánh hướng sang vùng gò Nguyệt.\n\n**Luận tướng:** Có vận số đi xa, thích hợp phát triển sự nghiệp hoặc định cư ở nước ngoài, nơi đất khách.")
            else:
                st.markdown("**Hình học:** Độ dài trung bình hoặc hơi ngắn.\n\n**Hướng đi:** Đường chạy dốc xuống, biên độ ôm gò ngón cái hẹp.\n\n**Luận tướng:** Cuộc sống thích hướng tới sự ổn định, an toàn, cần chú ý phân bổ thời gian làm việc để tránh lao lực.")

            # Tâm đạo
            st.subheader("2. 🔴 ĐƯỜNG TÂM ĐẠO (Tình Duyên & Cảm Xúc)")
            heart = classified.get('Heart_Line')
            if heart == 'Extreme_Curved_Passionate':
                st.markdown("**Hình học:** Đường siêu cong, uốn lượn mạnh mẽ.\n\n**Hướng đi:** Xuất phát từ rìa tay dưới ngón út, uốn một đường cong gắt hướng ngược lên phía kẽ ngón tay trỏ và ngón giữa.\n\n**Luận tướng:** Sống mãnh liệt, đặt nặng chuyện tình cảm, yêu ghét rõ ràng và có xu hướng che chở, chiếm hữu cao trong tình yêu.")
            else:
                st.markdown("**Hình học:** Đường thẳng hoặc có độ dài vừa phải.\n\n**Hướng đi:** Đường chạy cắt ngang lòng bàn tay và kết thúc ở gò dưới ngón giữa hoặc ngón trỏ.\n\n**Luận tướng:** Thuộc mẫu người lý trí, kiểm soát cảm xúc tốt, luôn giải quyết các xung đột tình cảm bằng cái đầu lạnh.")

            # Trí đạo
            st.subheader("3. 🔵 ĐƯỜNG TRÍ ĐẠO (Tư Duy & Sự Nghiệp)")
            head = classified.get('Head_Line')
            if head == 'Long_Curved_Creative':
                st.markdown("**Hình học:** Đường dài, võng sâu xuôi xuống.\n\n**Hướng đi:** Khởi hành từ kẽ tay trỏ, kéo dài và uốn cong chúc hẳn xuống khu vực đáy lòng bàn tay (gò Nguyệt).\n\n**Luận tướng:** Trí tưởng tượng bay bổng, tư duy sáng tạo vượt trội, rất nhạy cảm với nghệ thuật, ngôn từ hoặc thiết kế.")
            else:
                st.markdown("**Hình học:** Đường thẳng, độ dài trung bình dứt khoát.\n\n**Hướng đi:** Chạy ngang phẳng qua lòng bàn tay, dừng lại ở khoảng dưới ngón áp út.\n\n**Luận tướng:** Tư duy thực tế, logic, khả năng giải quyết vấn đề và ứng biến với áp lực công việc vô cùng nhanh nhạy.")
        else:
            st.warning("⚠️ Không nhận diện được bàn tay. Hãy giữ bàn tay tĩnh và rõ ràng trước ống kính camera nhé!")
