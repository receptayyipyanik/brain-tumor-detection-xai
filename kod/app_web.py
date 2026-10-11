"""
Beyin Tümörü Tespiti & Açıklanabilir Yapay Zeka (Grad-CAM) Web Arayüzü
Gradio kullanarak tek tıkla yerel veya Hugging Face Spaces üzerinde çalıştırılabilir.
Kullanım:
    pip install gradio
    python kod/app_web.py
"""
import os
import tempfile
import gradio as gr
from PIL import Image
from inference import predict_and_explain

def process_image(input_img):
    if input_img is None:
        return "Lütfen bir MR görseli yükleyin.", None
    
    # Geçici dosyaya kaydedip inference fonksiyonuna veriyoruz
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        input_img.save(tmp.name)
        tmp_path = tmp.name

    try:
        res = predict_and_explain(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if "error" in res:
        return f"HATA: {res['error']}", None

    if not res["has_tumor"]:
        durum_metni = f"✅ SAĞLIKLI (Tümör Bulunamadı)\nGüven Oranı: %{res.get('confidence', 0):.2f}"
        return durum_metni, None

    tumor_turu = res.get("tumor_type", "Bilinmiyor").upper()
    tumor_conf = res.get("tumor_confidence", 0)
    genel_conf = res.get("confidence", 0)
    durum_metni = (
        f"⚠️ TÜMÖR TESPİT EDİLDİ!\n"
        f"• Tümör Varlığı Güveni: %{genel_conf:.2f}\n"
        f"• Tahmin Edilen Tür: {tumor_turu}\n"
        f"• Tür Sınıflandırma Güveni: %{tumor_conf:.2f}"
    )

    heatmap_img = None
    if res.get("heatmap_path") and os.path.exists(res["heatmap_path"]):
        heatmap_img = Image.open(res["heatmap_path"])

    return durum_metni, heatmap_img

demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="MR Görüntüsü Yükleyin (JPG/PNG)"),
    outputs=[
        gr.Textbox(label="Teşhis ve Güven Analizi", lines=4),
        gr.Image(type="pil", label="Grad-CAM Isı Haritası (Odaklanılan Bölge)")
    ],
    title="🧠 Beyin Tümörü Tespiti & Açıklanabilir AI (Grad-CAM)",
    description="ResNet18 + CBAM Dikkat Mekanizması ve Grad-CAM Açıklanabilirlik Modeli ile Beyin MR Analizi.",
    examples=[]
)

if __name__ == "__main__":
    demo.launch()
