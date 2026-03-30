# 🧪 Prompt Eval App

![Prompt Eval App](image.png)

แอปพลิเคชันสำหรับประเมินผล (Evaluate) Prompt และความแม่นยำของคำตอบจาก LLMs สร้างด้วย [Streamlit](https://streamlit.io/) และเชื่อมต่อกับ LLMs ผ่าน API ของ [OpenRouter](https://openrouter.ai/)

## 🌟 ฟีเจอร์หลัก (Features)

- **อัปโหลด Dataset**: รองรับการนำเข้าไฟล์ CSV สำหรับทดสอบ Prompt ทีละหลายๆ ข้อ (Batch Processing)
- **รองรับหลากหลายโมเดล**: สามารถเปลี่ยนโมเดลหน้า UI ได้ทันที (เช่น `anthropic/claude-3.5-sonnet`, `openai/gpt-4o`, `google/gemini-2.0-flash`)
- **รูปแบบการประเมิน (Graders) 4 ประเภท**:
  1. 📘 `exact_match`: ข้อความคำตอบและสิ่งที่คาดหวังตรงกัน 100%
  2. 🪻 `rule_based`: ประเมินตามกฎที่กำหนดไว้ล่วงหน้า เช่น การมีอยู่ของคำ/คีย์ที่ต้องการ (`required_keys`), ชนิดภาษา หรือจำกัดจำนวนประโยคสูงสุด
  3. 📗 `llm_judge`: ใช้ LLM เป็นผู้ติดสิน (Judge) ให้คะแนนความสมเหตุสมผลของระบุตั้งแต่ 0.0 - 1.0
  4. 📙 `tool_use`: ตรวจสอบการเรียกใช้งานเครื่องมือ (Tool Calling) ว่าเรียกถูกฟังก์ชันไหม และส่ง parameter มาครบ/ตรงตามคาดหวังหรือไม่
- **รายงานผลและ Export**: สรุประดับคะแนนเฉลี่ย เกณฑ์ผ่าน/ไม่ผ่าน (Pass Rate) แยกรายประเภท Grader และดาวน์โหลดผลลัพธ์กลับเป็น CSV ได้
- **ปรับแต่งได้ (Customization)**: สามารถปรับแต่ง *System Prompt* และ *Pass Threshold* หน้าเว็บได้เลย

## 📦 การติดตั้งและการรันโปรเจกต์ (Installation)

โปรเจกต์นี้ระบุ Version ว่าต้องใช้ **Python 3.10 ขึ้นไป** ภายในโปรเจกต์มีการใช้ `uv` สำหรับจัดการ package (มี `uv.lock`, `pyproject.toml`)

**1. Clone โปรเจกต์**
```bash
git clone <repository-url>
cd prompt_eval_app
```

**2. ติดตั้ง Dependencies**
หากคุณติดตัังและใช้งาน `uv` อยู่ สามารถสั่งซิงก์ dependencies ได้เลย:
```bash
uv sync
```
*(ถ้าไม่ใช้ `uv` สามารถรันคำสั่ง `pip install streamlit python-dotenv pandas requests` ใน environment ของคุณแทนได้)*

**3. ตั้งค่า Environment Variables**
สร้างไฟล์ `.env` ไว้ในโฟลเดอร์เดียวกันกับตัวโปรเจกต์ และใส่ค่า Config:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
# OPENROUTER_MODEL=anthropic/claude-sonnet-4-5 (Optional, ระบุโมเดลเริ่มต้นได้)
```

**4. รันแอปพลิเคชัน**
```bash
streamlit run prompt_eval_app.py
```
*(หรือถ้ารันใน uv environment: `uv run streamlit run prompt_eval_app.py`)*

เมื่อรันเสร็จเรียบร้อย ให้เปิดหน้าเบราว์เซอร์ลัดไปที่ `http://localhost:8501`

## 📊 โครงสร้างข้อมูล Dataset (.CSV)

เพื่อให้ระบบรันประเมินผลได้ ไฟล์ CSV ที่นำมาอัปโหลดจะต้องมี Schema โครงสร้างอย่างน้อยดังนี้ (ระบบสามารถรองรับ Encoding ภาษาไทยได้):

- `id`: ลำดับของข้อมูล
- `input`: ข้อความ/คำถามที่จะให้ AI ประมวลผล
- `expected`: คำตอบที่คาดหวัง (ถ้ารูปแบบเป็น Rule-based หรือ Tool-use ต้องกรอกให้อยู่ในรูปแบบ JSON payload)
- `grader`: ชนิดรูปแบบการประเมิน (`exact_match`, `rule_based`, `llm_judge`, `tool_use`)

> 💡 **Tip:** คุณสามารถกดปุ่ม **⬇ ดาวน์โหลด template** เป็นไฟล์ CSV พื้นฐานในหน้า App ได้

## 🛠 เครื่องมือจำลอง (Mock Tools)
แอปพลิเคชันได้ Mock Tool เบื้องต้นเอาไว้สำหรับการประเมินประเภท `tool_use` ได้แก่:
- `get_tasks` - รับ param: `due` (ดึง tasks จาก Asana)
- `web_search` - รับ param: `query` (ค้นหาข้อมูลจากอินเทอร์เน็ต)
- `create_issue` - รับ param: `title` (สร้าง issue ใน Linear)
- `get_issues` - (ดึง issues ทั้งหมดจาก Linear)
