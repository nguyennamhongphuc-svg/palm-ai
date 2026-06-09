import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# Cấu hình giao diện Streamlit rộng rãi, đẹp mắt
st.set_page_config(page_title="AI Xem Chỉ Tay", page_icon="🔮", layout="centered")

st.title("🔮 AI ĐỌC CHỈ TAY CÔNG NGHỆ CAO (STREAMLIT APP)")
st.caption("Ứng dụng tự động phân tích độ cong, độ dài và hướng đi của các đường chỉ tay bằng AI.")

# ==========================================
# THUẬT TOÁN TOÁN HỌC VẼ ĐƯỜNG CONG BEZIER
# ==========================================
def draw_quadratic_bezier(img, p0, p1, p2, color, thickness=4):
    """Vẽ đường cong bậc 2 qua 3 điểm điều hướng"""
    t_values = np.linspace(0, 1, 30)
    points = []
    for t in t_values:
        x = int((1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0])
        y = int((1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1])
        points.append((x, y))
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], color, thickness)

def draw_cubic_bezier(img, p0, p1, p2, p3, color, thickness=4):
    """Vẽ đường cong bậc 3 qua 4 điểm điều hướng (Dành cho Sinh Đạo)"""
    t_values = np.linspace(0, 1, 30)
    points = []
    for t in t_values:
        x = int((1 - t)**3 * p0[0] + 3 * (1 - t)**2 * t * p1[0] + 3 * (1 - t) * t**2 * p2[0] + t**3 * p3[0])
        y = int((1 - t)**3 * p0[1] + 3 * (1 - t)**2 * t * p1[1] + 3 * (1 - t) * t**2 * p2[1] + t**3 * p3[1])
        points.append((x, y))
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], color, thickness)

# ==========================================
# BƯỚC 3: HÀM PHÂN TÍCH HÌNH HỌC (OPENCV)
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
            else:
                curvature = 1.0
            features[key] = {'length': length, 'curvature': curvature, 'clarity': clarity}
        else:
            features[key] = {'length': 0.0, 'curvature': 1.0, 'clarity': 0.0}
    return features

def run_rule_engine(features):
    results = {}
    # Sinh đạo
    life_len = features['Life_Line']['length']
    life_clar = features['Life_Line']['clarity']
    life_curv = features['Life_Line']['curvature']
    if life_len > 380: results['Life_Line'] = 'Super_Long_Vigorous' if life_clar > 70 else 'Long_Branch_Out'
    elif life_len > 220: results['Life_Line'] = 'Medium_Curved_Active' if life_curv > 1.35 else 'Medium_Straight_Stable'
    else: results['Life_Line'] = 'Short_Deep_Dense' if life_clar > 80 else 'Short_Fragile'

    # Tâm đạo
    heart_curv = features['Heart_Line']['curvature']
    heart_len = features['Heart_Line']['length']
    if heart_curv > 1.55: results['Heart_Line'] = 'Extreme_Curved_Passionate'
    elif heart_curv > 1.15: results['Heart_Line'] = 'Straight_Long_Altruist' if heart_len > 280 else 'Straight_Short_Cold'
    else: results['Heart_Line'] = 'Chain_Broken_Anxious'

    # Trí đạo
    head_clar = features['Head_Line']['clarity']
    head_len = features['Head_Line']['length']
    head_curv = features['Head_Line']['curvature']
    if head_len > 320: results['Head_Line'] = 'Long_Curved_Creative' if head_curv > 1.25 else 'Long_Straight_Mastermind'
    elif head_len > 180: results['Head_Line'] = 'Medium_Balanced_Adaptable'
    else: results['Short_Material_Fast'] if head_clar > 75 else 'Short_Faint_Distracted'
    return results

# ==========================================
# BƯỚC 4: GIAO DIỆN CAMERA & HIỂN THỊ STREAMLIT
# ==========================================
camera_image = st.camera_input("Xòe rộng lòng bàn tay của bạn trước camera và bấm Chụp:")

if camera_image is not None:
    # Đọc ảnh từ Streamlit sang định dạng OpenCV
    file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_draw = img_bgr.copy() # Bản để vẽ đè đường cong lên
    h, w, _ = img_bgr.shape

    # Gọi MediaPipe Hands bản Python
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, model_complexity=1) as hands:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            st.success("✅ AI đã nhận diện được khung xương tay! Đang khớp tọa độ đường cong...")
            
            landmarks = results.multi_hand_landmarks[0].landmark
            
            # Chuyển đổi tọa độ pixel
            def get_pt(lm): return (int(lm.x * w), int(lm.y * h))
            
            wrist = get_pt(landmarks[0])
            thumb_cmc = get_pt(landmarks[1])
            index_mcp = get_pt(landmarks[5])
            middle_mcp = get_pt(landmarks[9])
            ring_mcp = get_pt(landmarks[13])
            pinky_mcp = get_pt(landmarks[17])

            center = (int((wrist[0] + middle_mcp[0]) / 2), int((wrist[1] + middle_mcp[1]) / 2))
            start_joint = (int((thumb_cmc[0] + index_mcp[0]) / 2), int((thumb_cmc[1] + index_mcp[1]) / 2))

            # 1. Vẽ ĐƯỜNG TÂM ĐẠO (Màu đỏ) theo tỷ lệ cơ thể
            heart_start = (int(pinky_mcp[0] * 0.8 + wrist[0] * 0.2), int(pinky_mcp[1] * 0.8 + wrist[1] * 0.2))
            heart_end = (int((index_mcp[0] + middle_mcp[0]) / 2), int((index_mcp[1] + middle_mcp[1]) / 2))
            heart_control = (int(ring_mcp[0] * 0.7 + wrist[0] * 0.3), int(ring_mcp[1] * 0.7 + wrist[1] * 0.3))
            draw_quadratic_bezier(img_draw, heart_start, heart_control, heart_end, (0, 0, 255), 5) # Màu Đỏ trong BGR

            # 2. Vẽ ĐƯỜNG TRÍ ĐẠO (Màu xanh dương)
            head_end = (int(pinky_mcp[0] * 0.5 + wrist[0] * 0.5), int(pinky_mcp[1] * 0.5 + wrist[1] * 0.5))
            draw_quadratic_bezier(img_draw, start_joint, center, head_end, (255, 0, 0), 5) # Màu Xanh Dương trong BGR

            # 3. Vẽ ĐƯỜNG SINH ĐẠO (Màu xanh lá)
            life_control2 = (int(thumb_cmc[0] * 0.4 + wrist[0] * 0.6), int(thumb_cmc[1] * 0.4 + wrist[1] * 0.6))
            draw_cubic_bezier(img_draw, start_joint, center, life_control2, wrist, (0, 255, 0), 5) # Màu Xanh Lá trong BGR

            # Hiển thị ảnh sau khi vẽ đường cong lên Streamlit
            st.image(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB), caption="Mô phỏng độ cong chỉ tay của bạn", use_container_width=True)

            # Chạy trích xuất đặc trưng Canny gốc và luận giải
            features = extract_line_features(img_bgr)
            classified = run_rule_engine(features)

            # --- KHU VỰC IN KẾT QUẢ ĐÃ ĐƯỢC ĐỊNH DẠNG ĐẸP ---
            st.markdown("## 📊 KẾT QUẢ PHÂN TÍCH CHI TIẾT")
            
            # Khối Sinh Đạo
            st.subheader("1. 🟢 ĐƯỜNG SINH ĐẠO (Vận Mệnh & Sức Khỏe)")
            life = classified.get('Life_Line')
            if life == 'Super_Long_Vigorous':
                st.markdown("**Dạng hình học:** Đường rất dài, nét đậm sâu. \n\n**Hướng cong:** Lượn vòng cung lớn, ôm trọn gò Kim Tinh xuống sát cổ tay. \n\n**Luận tướng:** Năng lượng dồi dào, sức khỏe bền bỉ dẻo dai.")
            elif life == 'Long_Branch_Out':
                st.markdown("**Dạng hình học:** Đường dài, có nhánh chẻ đôi ở cuối đuôi. \n\n**Hướng cong:** Chạy hướng về phía cổ tay nhưng rẽ nhánh rõ rệt sang hướng gò Nguyệt. \n\n**Luận tướng:** Có số định cư xa quê hương, hậu vận đi lại nhiều hoặc đổi nơi ở.")
            elif life == 'Medium_Curved_Active':
                st.markdown("**Dạng hình học:** Chiều dài trung bình, bản rộng. \n\n**Hướng cong:** Bo cong mềm mại, mở rộng biên độ ra giữa lòng bàn tay. \n\n**Luận tướng:** Tính cách hướng ngoại, năng động, thích nghi rất tốt với xã hội.")
            else:
                st.markdown("**Dạng hình học:** Đường ngắn hoặc hơi mảnh. \n\n**Hướng cong:** Chạy dốc xuống, độ cong ít ôm sát ngón cái. \n\n**Luận tướng:** Sức khỏe ở mức bình thường, cần chú ý nghỉ ngơi điều độ tránh làm việc quá sức.")

            # Khối Tâm Đạo
            st.subheader("2. 🔴 ĐƯỜNG TÂM ĐẠO (Tình Duyên & Cảm Xúc)")
            heart = classified.get('Heart_Line')
            if heart == 'Extreme_Curved_Passionate':
                st.markdown("**Dạng hình học:** Đường siêu cong, uốn ngược sắc nét. \n\n**Hướng cong:** Vuốt cong gắt từ rìa ngón út chạy hướng thẳng lên kẽ giữa ngón trỏ và ngón giữa. \n\n**Luận tướng:** Người giàu cảm xúc, yêu ghét phân minh rõ ràng. Sống mãnh liệt nhưng đôi khi hơi chiếm hữu.")
            elif heart == 'Straight_Long_Altruist':
                st.markdown("**Dạng hình học:** Đường thẳng băng, kéo dài trường nét. \n\n**Hướng cong:** Cắt ngang phẳng lặng xuyên suốt đến tận vùng gò dưới ngón trỏ. \n\n**Luận tướng:** Người lý trí cao trong tình cảm, bao dung, biết hy sinh nhưng ít khi bộc lộ cảm xúc ra ngoài.")
            else:
                st.markdown("**Dạng hình học:** Đường ngắn, kết thúc sớm hoặc lượn sóng nhẹ. \n\n**Hướng cong:** Kết thúc ngay dưới khu vực ngón giữa. \n\n**Luận tướng:** Tâm lý thực tế, xử lý mọi chuyện bằng lý trí, đôi khi tỏ ra hơi lạnh lùng.")

            # Khối Trí Đạo
            st.subheader("3. 🔵 ĐƯỜNG TRÍ ĐẠO (Tư Duy & Sự Nghiệp)")
            head = classified.get('Head_Line')
            if head == 'Long_Curved_Creative':
                st.markdown("**Dạng hình học:** Đường dài, võng cong mềm mại. \n\n**Hướng cong:** Xuất phát từ cạnh tay, chạy dài và uốn cong hẳn xuống gò Nguyệt (đáy lòng bàn tay). \n\n**Luận tướng:** Đầu óc sáng tạo cực tốt, có thiên hướng nghệ thuật, trực giác và trí tưởng tượng phong phú.")
            elif head == 'Long_Straight_Mastermind':
                st.markdown("**Dạng hình học:** Đường dài, thẳng tắp như kẻ chỉ. \n\n**Hướng cong:** Cắt đôi ngang lòng bàn tay, chạy song song với đường tình duyên. \n\n**Luận tướng:** Tư duy logic vượt trội, phân tích dữ liệu sắc bén, là mẫu người mưu lược.")
            else:
                st.markdown("**Dạng hình học:** Chiều dài trung bình, hơi chếch nhẹ. \n\n**Hướng cong:** Điểm cuối dừng ở khoảng dưới ngón áp út. \n\n**Luận tướng:** Trí tuệ thực tiễn, khả năng ứng biến, xoay chuyển tình huống trong công việc rất nhanh chóng.")

        else:
            st.warning("⚠️ Không tìm thấy bàn tay trong ảnh chụp! Bạn hãy để tay thẳng thắn, rõ ràng trước camera rồi bấm chụp lại nhé.")
