using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace TumorUI;

public partial class Form1 : Form
{
    private string secilenGoruntuYolu = "";


    private Button btnYukle;
    private Button btnSorgula;
    private PictureBox picOrijinal;
    private PictureBox picIsıHaritasi;
    private Label lblSonuc;

    private Button btnAyarlar;

    public Form1()
    {
        InitializeComponent();
    }

    protected override void OnLoad(EventArgs e)
    {
        base.OnLoad(e);
        SetupUI();
    }

    private void SetupUI()
    {
        this.Text = "Beyin Tümörü Tespiti (C# & Python Entegrasyonu)";
        this.BackColor = Color.FromArgb(30, 30, 30);
        this.WindowState = FormWindowState.Maximized;

        Panel pnlTop = new Panel() { Dock = DockStyle.Top, Height = 120 };
        
        btnYukle = new Button() { Text = "Görsel Yükle", Location = new Point(40, 25), Size = new Size(160, 45), BackColor = Color.FromArgb(0, 150, 136), ForeColor = Color.White, Font = new Font("Segoe UI", 12, FontStyle.Bold), FlatStyle = FlatStyle.Flat, Cursor = Cursors.Hand };
        btnYukle.FlatAppearance.BorderSize = 0;
        btnYukle.Click += BtnLoad_Click;

        btnSorgula = new Button() { Text = "Sorgula", Location = new Point(220, 25), Size = new Size(160, 45), BackColor = Color.FromArgb(41, 121, 255), ForeColor = Color.White, Font = new Font("Segoe UI", 12, FontStyle.Bold), FlatStyle = FlatStyle.Flat, Cursor = Cursors.Hand };
        btnSorgula.FlatAppearance.BorderSize = 0;
        btnSorgula.Click += BtnQuery_Click;

        btnAyarlar = new Button() { Text = "⚙️ Python Seç", Anchor = AnchorStyles.Top | AnchorStyles.Right, Location = new Point(this.ClientSize.Width - 200, 25), Size = new Size(160, 45), BackColor = Color.FromArgb(85, 85, 85), ForeColor = Color.White, Font = new Font("Segoe UI", 11, FontStyle.Bold), FlatStyle = FlatStyle.Flat, Cursor = Cursors.Hand };
        btnAyarlar.FlatAppearance.BorderSize = 0;
        btnAyarlar.Click += (s, e) => SetPythonPath();

        lblSonuc = new Label() { Text = "Lütfen bir MR görseli yükleyin.", Location = new Point(40, 85), Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right, AutoSize = false, Width = this.ClientSize.Width - 80, Height = 35, Font = new Font("Segoe UI", 14, FontStyle.Bold), ForeColor = Color.LightGray };

        pnlTop.Controls.Add(btnYukle);
        pnlTop.Controls.Add(btnSorgula);
        pnlTop.Controls.Add(btnAyarlar);
        pnlTop.Controls.Add(lblSonuc);

        TableLayoutPanel tlpImages = new TableLayoutPanel() { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 2 };
        tlpImages.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50F));
        tlpImages.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50F));
        tlpImages.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));
        tlpImages.RowStyles.Add(new RowStyle(SizeType.Absolute, 40F));

        picOrijinal = new PictureBox() { Dock = DockStyle.Fill, Margin = new Padding(40, 20, 20, 20), SizeMode = PictureBoxSizeMode.Zoom, BorderStyle = BorderStyle.None, BackColor = Color.FromArgb(45, 45, 48) };
        picIsıHaritasi = new PictureBox() { Dock = DockStyle.Fill, Margin = new Padding(20, 20, 40, 20), SizeMode = PictureBoxSizeMode.Zoom, BorderStyle = BorderStyle.None, BackColor = Color.FromArgb(45, 45, 48) };

        Label lblOrig = new Label() { Text = "Orijinal Görüntü", Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleCenter, Font = new Font("Segoe UI", 12, FontStyle.Regular), ForeColor = Color.DarkGray };
        Label lblHeat = new Label() { Text = "Isı Haritası (Grad-CAM)", Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleCenter, Font = new Font("Segoe UI", 12, FontStyle.Regular), ForeColor = Color.DarkGray };

        tlpImages.Controls.Add(picOrijinal, 0, 0);
        tlpImages.Controls.Add(picIsıHaritasi, 1, 0);
        tlpImages.Controls.Add(lblOrig, 0, 1);
        tlpImages.Controls.Add(lblHeat, 1, 1);

        this.Controls.Add(tlpImages);
        this.Controls.Add(pnlTop);
        tlpImages.BringToFront();
    }

    private void BtnLoad_Click(object? sender, EventArgs e)
    {
        using (OpenFileDialog ofd = new OpenFileDialog())
        {
            ofd.Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp";
            ofd.Title = "MR Görüntüsü Seçin";

            if (ofd.ShowDialog(this) == DialogResult.OK)
            {
                secilenGoruntuYolu = ofd.FileName;


                using (var fs = new FileStream(secilenGoruntuYolu, FileMode.Open, FileAccess.Read))
                {
                    picOrijinal.Image = Image.FromStream(fs);
                }

                if (picIsıHaritasi.Image != null)
                {
                    picIsıHaritasi.Image.Dispose();
                    picIsıHaritasi.Image = null;
                }

                lblSonuc.Text = "Görsel yüklendi. Sorgulamaya hazır.";
                lblSonuc.ForeColor = Color.White;
            }
        }
    }

    private async void BtnQuery_Click(object? sender, EventArgs e)
    {
        if (string.IsNullOrEmpty(secilenGoruntuYolu))
        {
            MessageBox.Show("Lütfen önce bir görsel yükleyin.", "Uyarı", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }

        lblSonuc.Text = "Model analiz ediyor, lütfen bekleyin...";
        lblSonuc.ForeColor = Color.DeepSkyBlue;
        btnSorgula.Enabled = false;

        try
        {
            await Task.Run(() => RunInference(secilenGoruntuYolu));
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Analiz sırasında bir hata oluştu:\n{ex.Message}", "Hata", MessageBoxButtons.OK, MessageBoxIcon.Error);
            lblSonuc.Text = "Analiz başarısız oldu.";
            lblSonuc.ForeColor = Color.IndianRed;
        }
        finally
        {
            btnSorgula.Enabled = true;
        }
    }

    private void RunInference(string goruntuYolu)
    {

        string temelYol = AppDomain.CurrentDomain.BaseDirectory;
        string pythonProjeKlasoru = Path.GetFullPath(Path.Combine(temelYol, @"..\..\..\..\"));
        string scriptYolu = Path.Combine(pythonProjeKlasoru, "inference.py");

        if (!File.Exists(scriptYolu))
        {

            pythonProjeKlasoru = Path.GetFullPath(Path.Combine(temelYol, @"..\"));
            scriptYolu = Path.Combine(pythonProjeKlasoru, "inference.py");

            if (!File.Exists(scriptYolu))
            {

                pythonProjeKlasoru = temelYol;
                scriptYolu = Path.Combine(pythonProjeKlasoru, "inference.py");
            }
        }

        if (!File.Exists(scriptYolu))
        {
            throw new FileNotFoundException($"Python script dosyası bulunamadı: {scriptYolu}");
        }

        ProcessStartInfo psiObj = new ProcessStartInfo
        {
            FileName = GetPythonPath(),
            Arguments = $"\"{scriptYolu}\" --image \"{goruntuYolu}\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = pythonProjeKlasoru
        };

        using (Process islem = new Process())
        {
            islem.StartInfo = psiObj;
            try
            {
                islem.Start();
            }
            catch (System.ComponentModel.Win32Exception)
            {
                // Python bulunamadıysa
                this.Invoke(new Action(() => SetPythonPath()));
                return;
            }

            string ciktiJson = islem.StandardOutput.ReadToEnd();
            string hataCikti = islem.StandardError.ReadToEnd();
            islem.WaitForExit();

            if (islem.ExitCode != 0)
            {
                string errMsg = string.IsNullOrWhiteSpace(hataCikti) ? ciktiJson : hataCikti;
                this.Invoke(new Action(() => 
                {
                    MessageBox.Show($"Python Script Hatası (Exit Code: {islem.ExitCode}):\n{errMsg}", "Python Hatası", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    lblSonuc.Text = "Analiz başarısız oldu.";
                    lblSonuc.ForeColor = Color.IndianRed;
                }));
                return;
            }

            // Çıktıda hata veya loglar olabilir. JSON formatını parse et:
            try
            {
                string[] lines = ciktiJson.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
                string jsonLine = lines[lines.Length - 1];

                var jsonDoc = JsonDocument.Parse(jsonLine);
                var root = jsonDoc.RootElement;

                if (root.TryGetProperty("error", out JsonElement errorElement))
                {
                    string pyError = errorElement.GetString();
                    this.Invoke(new Action(() => 
                    {
                        MessageBox.Show(pyError, "Python Model Hatası", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        lblSonuc.Text = "Eksik model dosyası.";
                        lblSonuc.ForeColor = Color.Orange;
                    }));
                    return;
                }

                bool tumorVar = root.GetProperty("has_tumor").GetBoolean();

                this.Invoke(new Action(() =>
                {
                    if (tumorVar)
                    {
                        string tumorTur = root.GetProperty("tumor_type").GetString();
                        string isiHaritaYolu = root.GetProperty("heatmap_path").GetString();

                        lblSonuc.Text = $"TÜMÖR TESPİT EDİLDİ! Tür: {tumorTur}";
                        lblSonuc.ForeColor = Color.IndianRed;

                        if (File.Exists(isiHaritaYolu))
                        {
                            using (var fs = new FileStream(isiHaritaYolu, FileMode.Open, FileAccess.Read))
                            {
                                if (picIsıHaritasi.Image != null) picIsıHaritasi.Image.Dispose();
                                picIsıHaritasi.Image = Image.FromStream(fs);
                            }
                        }
                    }
                    else
                    {
                        lblSonuc.Text = "SAĞLIKLI (Tümör Bulunamadı)";
                        lblSonuc.ForeColor = Color.LightGreen;
                        if (picIsıHaritasi.Image != null) picIsıHaritasi.Image.Dispose();
                        picIsıHaritasi.Image = null;
                    }
                }));
            }
            catch (JsonException ex)
            {
                throw new Exception($"Python'dan geçersiz bir yanıt geldi.\nDetay: {ex.Message}\nÇıktı: {ciktiJson}");
            }
        }
    }

    private string GetPythonPath()
    {
        string configPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python_path.txt");
        if (File.Exists(configPath))
        {
            return File.ReadAllText(configPath).Trim();
        }
        return "python";
    }

    private void SetPythonPath()
    {
        MessageBox.Show("Sistemde Python bulunamadı. Lütfen Anaconda klasörünüzdeki veya sisteminizdeki 'python.exe' dosyasını seçin.", "Python Bulunamadı", MessageBoxButtons.OK, MessageBoxIcon.Information);
        using (OpenFileDialog ofd = new OpenFileDialog())
        {
            ofd.Filter = "Python Executable|python.exe";
            ofd.Title = "python.exe Dosyasını Seçin";
            if (ofd.ShowDialog(this) == DialogResult.OK)
            {
                string configPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python_path.txt");
                File.WriteAllText(configPath, ofd.FileName);
                MessageBox.Show("Python yolu kaydedildi! Sorgulamayı tekrar deneyebilirsiniz.", "Başarılı", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
        }
    }
}
