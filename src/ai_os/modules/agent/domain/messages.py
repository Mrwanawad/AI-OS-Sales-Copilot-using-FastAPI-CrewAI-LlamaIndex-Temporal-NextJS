import re

_ARABIC = re.compile(r"[؀-ۿ]")


def _is_arabic(text: str) -> bool:
    return _ARABIC.search(text) is not None


def escalated(question: str, ticket_id: str) -> str:
    if _is_arabic(question):
        return (
            "ليس لدي إجابة مؤكدة على هذا السؤال. "
            f"تم تحويله إلى قائد الفريق (رقم التذكرة {ticket_id})."
        )
    return (
        f"I don't have a verified answer to this. I sent it to your team lead (ticket {ticket_id})."
    )


def use_role_play(question: str) -> str:
    if _is_arabic(question):
        return "للتدريب، ابدأ جلسة تمثيل أدوار واختر شخصية المشتري."
    return "To practice, start a role-play session and pick a buyer persona."


def missing_details(question: str, missing: list[str]) -> str:
    listed = ", ".join(missing)
    if _is_arabic(question):
        return f"أحتاج إلى هذه البيانات لإجراء الحساب: {listed}."
    return f"I need these details to calculate this: {listed}."
