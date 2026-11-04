from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from shutil import copy2
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt
from docx.text.paragraph import Paragraph

ROOT = Path.cwd()
REPORT = next(p for p in ROOT.glob("*.docx") if p.stat().st_size == 263315 and "pre_chapter5" not in p.name)
BACKUP = next(p for p in ROOT.glob("*.docx") if "pre_chapter5_rewrite.bak" in p.name)
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def text_of(element):
    return "".join((node.text or "") for node in element.iter(f"{{{W}}}t"))


def clone_para(doc, template, text, *, bold=False, warning=False):
    element = deepcopy(template._p)
    p = Paragraph(element, doc._body)
    ppr = p._p.pPr
    for child in list(p._p):
        if child is not ppr:
            p._p.remove(child)
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    if bold:
        run.bold = True
    if warning:
        ppr = p._p.get_or_add_pPr()
        shade = OxmlElement("w:shd")
        shade.set(f"{{{W}}}val", "clear")
        shade.set(f"{{{W}}}color", "auto")
        shade.set(f"{{{W}}}fill", "FFF2CC")
        ppr.append(shade)
        borders = OxmlElement("w:pBdr")
        left = OxmlElement("w:left")
        for key, val in (("val", "single"), ("sz", "18"), ("space", "6"), ("color", "C65911")):
            left.set(f"{{{W}}}{key}", val)
        borders.append(left)
        ppr.append(borders)
    return p._p


def build_table(doc, headers, rows, widths, font_size=7.5):
    if len(headers) != len(widths) or any(len(row) != len(headers) for row in rows):
        raise ValueError("Table header/row/width count mismatch")
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for ri, values in enumerate([headers, *rows]):
        row = table.rows[0] if ri == 0 else table.add_row()
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        if ri == 0:
            row._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        for ci, value in enumerate(values):
            cell = row.cells[ci]
            cell.width = Inches(widths[ci])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if ri == 0 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(str(value))
            r.font.name = "Times New Roman"
            r.font.size = Pt(font_size)
            r.bold = ri == 0
            if ri == 0:
                shade = OxmlElement("w:shd")
                shade.set(f"{{{W}}}fill", "D9EAF7")
                cell._tc.get_or_add_tcPr().append(shade)
    return table._tbl


def find_heading(doc, title):
    for p in doc.paragraphs:
        if p.style.name == "Heading 1" and p.text.strip().startswith(title):
            return p._p
    raise ValueError(f"Heading not found: {title}")


def signatures_outside_ch5(doc):
    paras = doc.paragraphs
    start = next(i for i,p in enumerate(paras) if p.style.name == "Heading 1" and p.text.strip() == "THỰC NGHIỆM VÀ ĐÁNH GIÁ")
    end = next(i for i,p in enumerate(paras) if i > start and p.style.name == "Heading 1" and p.text.strip().startswith("KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN"))
    p_before = [p.text for p in paras[:start]]
    p_after = [p.text for p in paras[end:]]
    tables = [tuple(tuple(cell.text for cell in row.cells) for row in t.rows) for t in doc.tables]
    return p_before, p_after, tables


def main():
    original = Document(REPORT)
    original_signatures = signatures_outside_ch5(original)
    body = original._element.body
    children = list(body)
    start_heading = find_heading(original, "THỰC NGHIỆM VÀ ĐÁNH GIÁ")
    end_heading = find_heading(original, "KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN")
    start_pos, end_pos = children.index(start_el), children.index(end_el)
    old_ch5 = children[start_pos + 1:end_pos]
        body.remove(el)

    paras = original.paragraphs
    def template(prefix):
        return next(p for p in paras if p.text.strip().startswith(prefix))
    templates = {
        "h2": template("Kết quả huấn luyện"),
        "h2_notes": template("Nhận xét"),
        "h3_k": template("Lựa chọn số cụm k"),
        "h3_compare": template("So sánh mô hình tự cài đặt và thư viện"),
        "h3_profile": template("Đặc trưng các cụm"),
        "h3_target": template("Đối chiếu hậu nghiệm"),
        "h3_medoid": template("Medoid tại"),
        "body": template("Ý nghĩa các chỉ số"),
        "caption": template("Bảng 1-3."),
    }

    blocks = []
    def para(value, style="body", *, bold=False, warning=False):
        blocks.append(("p", value, style, bold, warning))
    def caption(value):
        blocks.append(("caption", value))
    def table(headers, rows, widths, size=7.5):
        blocks.append(("table", headers, rows, widths, size))

    para("Kết quả huấn luyện", "h2")
    para("Các kết quả trong mục này là từ mô hình thư viện kmedoids trên 2.283 hồ sơ của tất cả nguồn trong file tổng hợp. Lượt quét khảo sát k=2–8 dùng PAM/BUILD, seed được truyền là 42. Hiện chưa có kết quả bản tự cài đặt trên toàn bộ dữ liệu để so sánh.")
    para("Lựa chọn số cụm k", "h3_k")
    para("Silhouette đánh giá độ phù hợp của hồ sơ với cụm được gán so với cụm gần nhất khác. Với hồ sơ i, a(i) là khoảng cách trung bình tới các hồ sơ cùng cụm; b(i) là khoảng cách trung bình nhỏ nhất tới một cụm khác. Silhouette được tính bằng s(i) = [b(i) − a(i)] / max[a(i), b(i)]. Giá trị trung bình càng cao thường cho thấy cụm tách biệt hơn theo khoảng cách đang xét.")
    para("Inertia là tổng khoảng cách từ từng hồ sơ đến medoid của cụm được gán. Inertia thường giảm khi tăng k, nên cần diễn giải cùng silhouette và kích thước cụm. Không so trực tiếp inertia Manhattan với Gower vì hai thang khoảng cách khác nhau.")
    caption("Bảng 13. Silhouette và inertia của mô hình thư viện theo k")
    table(["k", "Manhattan silhouette", "Manhattan inertia", "Gower silhouette", "Gower inertia"], [
        ["2", "0,157", "15.802,364", "0,189", "435,111"], ["3", "0,102", "14.786,262", "0,191", "399,135"],
        ["4", "0,098", "14.087,215", "0,177", "371,822"], ["5", "0,111", "13.536,181", "0,171", "349,067"],
        ["6", "0,087", "13.126,483", "0,170", "331,357"], ["7", "0,088", "12.768,695", "0,162", "318,942"],
        ["8", "0,114", "12.418,026", "0,171", "307,385"],
    ], [0.45, 1.25, 1.25, 1.2, 1.05], 7.2)
    para("Trong sweep thư viện, Manhattan đạt silhouette cao nhất tại k=2 (0,1568), Gower cao nhất tại k=3 (0,1911). Các giá trị đều dưới 0,20, nên độ tách cụm còn hạn chế. Đây là lượt khảo sát k; cấu hình so sánh cuối đã chốt là k=2 cho ba lượt dự kiến. Gower k=3 chỉ là cực đại sweep, không phải cấu hình so sánh cuối.")

    para("So sánh mô hình tự cài đặt và thư viện (cấu hình cuối k=2)", "h3_compare")
    para("So sánh công bằng cần dùng cùng 2.283 hồ sơ, ma trận Manhattan 25 cột và k=2. Bảng 14 mô tả phương pháp và trạng thái kết quả; không điền số liệu giả cho bản tự cài đặt khi chưa chạy trên toàn dữ liệu. Ví dụ 30 hồ sơ Chương 4 chỉ minh họa thuật toán.")
    caption("Bảng 14. Cấu hình và tình trạng kết quả của hai mô hình")
    table(["Nội dung", "Tự cài đặt", "Thư viện kmedoids"], [
        ["Thuật toán", "Alternate K-Medoids: gán cụm rồi cập nhật medoid trong từng cụm", "PAM"],
        ["Khởi tạo", "Ngẫu nhiên seed 42; ví dụ có thể chỉ định initial_medoids", "BUILD; không bảo đảm cùng khởi tạo ngẫu nhiên"],
        ["Đầu vào so sánh", "Ma trận Manhattan 25 cột", "Cùng ma trận Manhattan 25 cột"],
        ["Cấu hình cuối", "k=2, seed=42", "k=2; khởi tạo BUILD"],
        ["Artifact toàn dữ liệu", "Chưa có; chưa thể kết luận hơn/kém", "Sweep đã có; Manhattan k=2: silhouette 0,1568, inertia 15.802,364"],
        ["Gower", "Không dùng trong so sánh implementation hiện tại", "Đối chứng về khoảng cách; sweep có k=2 và k=3"],
        ["Số vòng lặp", "Có thuộc tính n_iter_", "Adapter không cung cấp số vòng thực; -1 không phải số vòng"],
    ], [1.05, 2.65, 2.95], 7.1)
    para("Chưa thể kết luận mô hình nào tốt hơn trên toàn bộ dữ liệu. Sau khi chạy bản tự cài đặt Manhattan k=2, bổ sung silhouette, inertia, kích thước cụm và medoid. Không xem Manhattan–Gower là so sánh hai implementation.", bold=True, warning=True)

    para("Hồ sơ cụm của mô hình thư viện", "h3_profile")
    para("Bảng 15–16 trình bày Manhattan k=2 và Gower k=2 theo cấu hình cuối. Bảng 17 bổ sung Gower k=3, có silhouette cao nhất trong sweep Gower. Tỷ lệ target=1 chỉ là mô tả hậu nghiệm.")
    headers=["Cụm","Số mẫu (%)","Tuổi TB","Huyết áp TB","Cholesterol TB","Nhịp tim TB","Oldpeak TB","Tỷ lệ target=1"]
    widths=[0.4,1.0,0.7,0.85,0.95,0.9,0.75,0.9]
    caption("Bảng 15. Hồ sơ cụm Manhattan của mô hình thư viện (k=2)")
    table(headers,[["0","1.502 (65,8%)","51,61","128,60","242,50","148,90","0,43","0,452"],["1","781 (34,2%)","57,57","137,98","248,01","126,81","1,69","0,695"]],widths,7.0)
    caption("Bảng 16. Hồ sơ cụm Gower của mô hình thư viện (k=2; cấu hình đối chứng cuối)")
    table(headers,[["0","1.618 (70,9%)","52,46","130,72","244,12","145,30","0,67","0,457"],["1","665 (29,1%)","56,53","134,46","245,03","131,73","1,32","0,725"]],widths,7.0)
    caption("Bảng 17. Hồ sơ cụm Gower của mô hình thư viện (k=3; silhouette cao nhất trong sweep)")
    table(headers,[["0","1.173 (51,4%)","52,20","130,70","239,38","143,65","0,69","0,493"],["1","609 (26,7%)","57,02","134,52","243,74","129,86","1,34","0,719"],["2","501 (21,9%)","52,95","131,13","256,88","149,90","0,66","0,411"]],widths,7.0)

    para("Đối chiếu hậu nghiệm với target", "h3_target")
    para("ARI đo mức đồng thuận giữa cụm và target sau điều chỉnh ngẫu nhiên. Purity là tỷ lệ mẫu thuộc lớp chiếm đa số trong từng cụm, gộp trên toàn tập. Purity không phải accuracy phân loại vì mô hình phân cụm không học hàm dự đoán target.")
    caption("Bảng 18. Đối chiếu hậu nghiệm giữa cụm và target (mô hình thư viện)")
    table(["Khoảng cách","k","ARI với target","Purity","Ghi chú"],[["Manhattan","2","0,0378","0,5983","Cấu hình chính; đồng thuận yếu"],["Gower","2","0,0355","0,5957","Đối chứng cuối"],["Gower","3","0,0261","0,5817","Silhouette cao nhất trong sweep"]],[1.0,0.4,1.1,0.9,3.2],7.2)
    para("Tỷ lệ target=1 toàn tập khoảng 0,535. Purity 0,5817–0,5983 cao hơn baseline lớp đa số khoảng 4,6–6,3 điểm phần trăm; đây không phải accuracy hay bằng chứng chẩn đoán. ARI gần 0 biểu thị đồng thuận điều chỉnh với target yếu, nhưng không có nghĩa cụm sai.")

    para("Medoid thư viện tại các cấu hình được báo cáo", "h3_medoid")
    para("Chỉ số medoid trong artifact bắt đầu từ 0; target ở hàng medoid là nhãn của hồ sơ đại diện, không phải nhãn chung của cả cụm.")
    caption("Bảng 19. Medoid của mô hình thư viện tại các cấu hình được báo cáo")
    table(["Khoảng cách","k","Cụm","Medoid index (0-based)","Tuổi","Huyết áp","Cholesterol","Nhịp tim","Oldpeak","target"],[["Manhattan","2","0","644","54","120","239","155","0,0","1"],["Manhattan","2","1","1200","57","140","239","120","2,0","1"],["Gower","2","0","2018","57","128","229","150","0,4","0"],["Gower","2","1","1335","58","137","232","124","1,4","1"],["Gower","3","0","833","50","130","218","143","0,5","1"],["Gower","3","1","1335","58","137","232","124","1,4","1"],["Gower","3","2","1788","51","130","256","149","0,5","1"]],[0.8,0.35,0.35,0.85,0.4,0.55,0.6,0.5,0.45,0.35],6.3)
    para("Không gán nhãn “nguy cơ cao” cho cụm chỉ dựa vào target của medoid; cần xem tỷ lệ target và hồ sơ toàn cụm. Đây không phải kết luận y khoa.")

    para("Nhận xét", "h2_notes")
    para("Trong sweep thư viện, Manhattan đạt silhouette cao nhất ở k=2 (0,1568), Gower ở k=3 (0,1911). Các giá trị thấp cho thấy độ tách cụm còn hạn chế; đây không phải so sánh hai implementation.")
    para("Ở k=2, Manhattan chia thành 1.502 và 781 hồ sơ; Gower chia thành 1.618 và 665. Gower k=3 chia thành 1.173, 609 và 501 hồ sơ. Khác biệt đặc trưng là mô tả thống kê, không chứng minh nhân quả hoặc nguy cơ lâm sàng.")
    para("ARI giữa Manhattan và Gower trong sweep là 0,248–0,424, cho thấy hai cách biểu diễn chỉ đồng thuận một phần. Không so trực tiếp inertia giữa hai độ đo.")
    para("ARI hậu nghiệm với target lần lượt là 0,0378 (Manhattan k=2), 0,0355 (Gower k=2) và 0,0261 (Gower k=3). Purity lần lượt là 0,5983, 0,5957 và 0,5817; purity không phải accuracy và target không dùng để fit.")
    para("Giới hạn quan trọng: chưa có kết quả bản tự cài đặt trên toàn bộ 2.283 hồ sơ. Ví dụ 30 hồ sơ Chương 4 chỉ minh họa thuật toán. Chưa thể kết luận bản tự cài đặt tốt hơn hay kém hơn thư viện; sau khi chạy custom Manhattan k=2 cần cập nhật Bảng 14.",bold=True,warn=True)
    para("Kết quả còn phụ thuộc harmonize, xử lý thiếu, biểu diễn đặc trưng và khởi tạo. Nếu tiền xử lý đổi, cần chạy lại thí nghiệm. Kết quả phân cụm không thay thế chẩn đoán hoặc khuyến nghị y khoa.")

    # Construct new chapter in a temporary copy with python-docx, preserving styles.
    with NamedTemporaryFile(suffix=".docx", prefix="ch5_build_", dir=ROOT, delete=False) as f:
        build_path = Path(f.name)
    try:
        build = Document(REPORT)
        build_paras = build.paragraphs
        build_start = next(i for i,x in enumerate(build_paras) if x.style.name == "Heading 1" and x.text.strip() == "THỰC NGHIỆM VÀ ĐÁNH GIÁ")
        build_end = next(i for i,x in enumerate(build_paras) if i > build_start and x.style.name == "Heading 1" and x.text.strip().startswith("KẾT LUẬN ВА HƯỚNG PHÁT TRIỂN"))
        build_body = build._element.body
        build_children = list(build_body)
        start_el, end_el = build_paras[build_start]._p, build_paras[build_end]._p
        start_pos, end_pos = build_children.index(start_el), build_children.index(end_el)
        for element in build_children[start_pos+1:end_pos]:
            build_body.remove(element)
        build_templates = {}
        for key,prefix in [("h2","Kết quả huấn luyện"),("h2notes","Nhận xét"),("h3k","Lựa chọn số cụm k"),("h3compare","So sánh mô hình tự cài đặt và thư viện"),("h3profile","Đặc trưng các cụm"),("h3target","Đối chiếu hậu nghiệm"),("h3medoid","Medoid tại"),("body","Ý nghĩa các chỉ số"),("caption","Bảng 1-3.")]:
            build_templates[key] = next(p for p in build.paragraphs if p.text.strip().startswith(prefix))
        new_blocks=[]
        for item in blocks:
            if item[0]=="p":
                _,value,key,bold,warning=item
                new_blocks.append(clone_para(build,build_templates[key],value,bold=bold,warning=warning))
            elif item[0]=="caption":
                _,value=item
                new_blocks.append(clone_para(build,build_templates["caption"],value))
            else:
                _,headers,rows,widths,size=item
                new_blocks.append(add_table(build,headers,rows,widths,font_size=size))
        for element in new_blocks:
            end_el.addprevious(element)
        build.save(build_path)
        check=Document(build_path)
        texts=[p.text for p in check.paragraphs]
        required=["Bảng 13.","Bảng 14.","Bảng 15.","Bảng 16.","Bảng 17.","Bảng 18.","Bảng 19.","Nhận xét"]
        missing=[x for x in required if not any(x in t for t in texts)]
        if missing: raise RuntimeError(f"Missing required content: {missing}")
        if not any("chưa có kết quả bản tự cài đặt" in t.lower() and "2.283 hồ sơ" in t for t in texts): raise RuntimeError("Missing pending custom-result note")
        with ZipFile(build_path) as z:
        # Validate that all Chapter 4 and Chapter 6+ paragraph strings remain present.
        check_text = [p.text for p in check.paragraphs]
        original_text = [p.text for p in original.paragraphs]
        for marker in ["4.4.3. Ví dụ minh họa bài toán", "4.5.1. Thư viện và cấu hình mô hình", "Kết quả đạt được", "Hạn chế của đề tài"]:
            if not any(marker in x for x in check_text):
                raise RuntimeError(f"Content outside Chapter 5 missing: {marker}")
        if [p.text for p in check.paragraphs[new_end:]] != [p.text for p in original.paragraphs[end_para_index:]]:
            raise RuntimeError("Text from Chapter 6 onward changed")
        copy2(REPORT,BACKUP)
        build_path.replace(REPORT)
    finally:
        if build_path.exists(): build_path.unlink()
    print("Chapter 5 rewrite saved and validated:",REPORT)