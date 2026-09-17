using Newtonsoft.Json.Linq;
using System;
using System.Drawing;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Windows.Forms;
using System.Xml.Linq;

namespace VisionUpper2
{
    public partial class Form1 : Form
    {
        private static readonly HttpClient client = new HttpClient();
        private Button btnDetect;
        private PictureBox pictureBox;
        private Label lblResult;

        public Form1()
        {
            InitializeComponent();
            SetupUI();
        }

        private void SetupUI()
        {
            this.Text = "视觉引导机械臂上位机";
            this.Size = new Size(1024, 720);
            this.StartPosition = FormStartPosition.CenterScreen;

            btnDetect = new Button();
            btnDetect.Text = "检测";
            btnDetect.Location = new Point(20, 20);
            btnDetect.Size = new Size(100, 40);
            btnDetect.Click += btnDetect_Click;
            this.Controls.Add(btnDetect);

            pictureBox = new PictureBox();
            pictureBox.Location = new Point(20, 80);
            pictureBox.Size = new Size(960, 500);
            pictureBox.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBox.BorderStyle = BorderStyle.FixedSingle;
            this.Controls.Add(pictureBox);

            lblResult = new Label();
            lblResult.Location = new Point(20, 600);
            lblResult.Size = new Size(960, 80);
            lblResult.Font = new Font("Consolas", 10);
            this.Controls.Add(lblResult);
        }

        private async void btnDetect_Click(object sender, EventArgs e)
        {
            string imagePath = @"D:\Vision_Machine\dataset\images\0000.png";
            if (!File.Exists(imagePath))
            {
                MessageBox.Show("找不到图片: " + imagePath);
                return;
            }

            byte[] imageBytes = File.ReadAllBytes(imagePath);

            var content = new MultipartFormDataContent();
            var imageContent = new ByteArrayContent(imageBytes);
            imageContent.Headers.ContentType =
                new System.Net.Http.Headers.MediaTypeHeaderValue("image/png");
            content.Add(imageContent, "file", "test.png");

            try
            {
                var response = await client.PostAsync("http://127.0.0.1:8000/detect", content);
                string jsonStr = await response.Content.ReadAsStringAsync();

                JObject json = JObject.Parse(jsonStr);
                int count = json["count"].Value<int>();
                JArray objects = (JArray)json["objects"];

                Bitmap bmp = new Bitmap(imagePath);
                using (Graphics g = Graphics.FromImage(bmp))
                {
                    StringBuilder sb = new StringBuilder();
                    foreach (var obj in objects)
                    {
                        var bbox = obj["bbox_2d"];
                        var pos3d = obj["position_3d"];
                        var conf = obj["confidence"].Value<double>();

                        float x1 = bbox[0].Value<float>();
                        float y1 = bbox[1].Value<float>();
                        float x2 = bbox[2].Value<float>();
                        float y2 = bbox[3].Value<float>();
                        g.DrawRectangle(Pens.Lime, x1, y1, x2 - x1, y2 - y1);

                        float u = obj["center_2d"][0].Value<float>();
                        float v = obj["center_2d"][1].Value<float>();
                        g.FillEllipse(Brushes.Red, u - 3, v - 3, 6, 6);

                        sb.AppendLine($"3D: ({pos3d[0]}, {pos3d[1]}, {pos3d[2]}) conf: {conf}");
                    }
                    lblResult.Text = $"检测到 {count} 个方块\n{sb}";
                }
                pictureBox.Image = bmp;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"请求失败: {ex.Message}");
            }
        }
    }
}