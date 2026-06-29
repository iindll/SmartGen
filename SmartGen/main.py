# اقرأ الـ main الجديد من outputs وحطه مكان القديم
import urllib.request

# حمّل الـ main المعدّل وحطه مكان القديم
with open('/content/SmartGen/SmartGen/main.py', 'r') as f:
    content = f.read()

# أضف الـ arguments الجديدة
content = content.replace(
    "parser.add_argument('--need_generate'",
    """parser.add_argument('--api_key', default=None, type=str, help='API key')
    parser.add_argument('--api_base', default=None, type=str, help='API base URL')
    parser.add_argument('--compress_only', default=False, type=str2bool, help='Compress only')
    parser.add_argument('--need_generate'"""
)

with open('/content/SmartGen/SmartGen/main.py', 'w') as f:
    f.write(content)

print("Done!")
