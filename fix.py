"""Auto-fix the 3-space indentation bug in gridwise.py"""
import re

with open("gridwise.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix: any line with exactly 3 leading spaces followed by "model = os.getenv"
fixed = re.sub(
    r"^   model = os\.getenv",
    "    model = os.getenv",
    content,
    flags=re.MULTILINE,
)

if fixed != content:
    with open("gridwise.py", "w", encoding="utf-8") as f:
        f.write(fixed)
    print("✅ FIXED! 3 spaces -> 4 spaces")
else:
    print("ℹ️ No change needed (already correct)")