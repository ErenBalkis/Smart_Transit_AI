import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. MODEL MİMARİSİ (CSRNet)
# ==========================================
class CSRNet(nn.Module):
    """
    Kalabalık sayımı için kullanılan Congested Scene Recognition Network (CSRNet) modeli.
    VGG-16 omurgasını özellik çıkarıcı olarak kullanır ve sonrasında 
    genişletilmiş evrişim (dilated convolution) katmanları ile yoğunluk haritası üretir.
    """
    def __init__(self):
        super(CSRNet, self).__init__()
        self.frontend = self._make_frontend()
        self.backend = self._make_backend()
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def _make_frontend(self):
        # VGG-16 önceden eğitilmiş ağırlıklarıyla yüklenir
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        features = list(vgg16.features.children())
        # Özellik çıkarımı için ilk 23 katman kullanılır
        frontend = nn.Sequential(*features[0:23])
        return frontend

    def _make_backend(self):
        # Genişletilmiş evrişim (dilated convolution) ile uzamsal çözünürlük korunarak
        # modelin algı alanı (receptive field) artırılır.
        backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )
        return backend

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

# ==========================================
# 2. YARDIMCI FONKSİYONLAR
# ==========================================
@st.cache_resource
def load_model():
    """
    Model ağırlıklarını CPU üzerinde çalışacak şekilde yükler.
    Hugging Face Spaces gibi CPU tabanlı ortamlarda sorunsuz çalışmasını sağlar.
    """
    model = CSRNet()
    try:
        # best_model.pth dosyasının var olduğu varsayılır, yoksa hata yakalanır
        model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.error("Model ağırlıkları (best_model.pth) bulunamadı. Lütfen dosyayı proje dizinine ekleyin.")
        st.stop()
    model.eval()
    return model

# Görüntü ön işleme adımları (ImageNet standartları)
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. STREAMLIT ARAYÜZÜ (UI)
# ==========================================
st.set_page_config(page_title="Kalabalık Analizi Sistemi", page_icon="🚌", layout="wide")

# Yan Menü (Sidebar) Bilgilendirmesi
with st.sidebar:
    st.header("📌 Proje Hakkında")
    st.info(
        "Bu uygulama, **Akıllı Şehir** konsepti kapsamında toplu taşıma "
        "duraklarındaki kalabalık oranını analiz eder.\n\n"
        "Eğer duraktaki kişi sayısı kapasiteyi (50 kişi) aşarsa, "
        "sistem otomatik olarak **Ek Sefer** uyarısı verir."
    )
    st.markdown("---")
    st.write("🔧 **Altyapı:** PyTorch & CSRNet")
    st.write("📊 **Yöntem:** Yoğunluk Haritası Regresyonu")

st.title("🚌 Toplu Taşıma Kalabalık Analizi ve Ek Sefer Yönetimi")
st.markdown("Durağın fotoğrafını yükleyin, yapay zeka kalabalığı analiz edip sefer kararını versin.")

uploaded_file = st.file_uploader("Bir durak fotoğrafı seçin (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Resmi yükle
    image = Image.open(uploaded_file).convert('RGB')
    
    # Model tahmini için hazırlık ve inferans
    with st.spinner('Kalabalık analiz ediliyor, lütfen bekleyin...'):
        model = load_model()
        input_tensor = data_transforms(image).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(input_tensor)
            
        pred_map = prediction.squeeze().numpy()
        # Tahmini kişi sayısını negatif değerleri engelleyerek hesaplama
        tahmini_kisi = max(0, int(np.sum(pred_map)))
        
    st.markdown("---")
    
    # Görüntüleri Yan Yana Gösterme
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Orijinal Görüntü")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("🔥 Yoğunluk (Isı) Haritası")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(pred_map, cmap='jet')
        ax.axis('off')
        st.pyplot(fig)
        
    st.markdown("---")
    
    # Sonuçları Gösterme
    st.subheader("📊 Analiz Sonucu")
    
    # Streamlit Metric bileşeni ile gösterim
    st.metric(label="Tahmini Bekleyen Kişi Sayısı", value=tahmini_kisi)
    
    # İş Mantığı (Threshold = 50)
    kapasite_siniri = 50
    if tahmini_kisi > kapasite_siniri:
        st.error(f"⚠️ **DİKKAT:** Durak Kapasitesi Aşıldı! (Sınır: {kapasite_siniri} Kişi)\n\nMerkeze **Ek Sefer** talebi iletiliyor.")
    else:
        st.success("✅ Yolcu sayısı normal seviyede. Ek bir işleme gerek yok.")