from ai_os.modules.coaching.domain.models import Persona

# Invented buyers. Practice never draws on real client data.
PERSONAS = {
    persona.id: persona
    for persona in (
        Persona(
            "first-time-buyer",
            "Salma, first-time buyer",
            "A young professional buying her first apartment on a tight budget. Careful and "
            "unsure how installment plans work.",
            (
                "The down payment is more than I saved",
                "I am afraid of committing to payments for years",
                "My family thinks I should wait",
            ),
            "Hi, I saw an ad for your compounds. I'm just looking, and I may not afford it.",
        ),
        Persona(
            "investor",
            "Karim, investor",
            "An experienced investor who owns two units and compares every deal on returns. "
            "Direct, short on time, and quick to dismiss sales talk.",
            (
                "Rental yield here looks weak",
                "Another developer offered better terms",
                "Resale in this area is slow",
            ),
            "I have ten minutes. Tell me why this beats what I can already buy elsewhere.",
        ),
        Persona(
            "family-upgrader",
            "Hana, upgrading for her family",
            "A mother of two moving to a bigger home. She cares about schools, safety, and "
            "whether the developer delivers on time.",
            (
                "Developers always deliver late",
                "I don't know what schools are nearby",
                "My husband has to agree before anything happens",
            ),
            "We need more space for the kids, but I've heard bad stories about late delivery.",
        ),
        Persona(
            "expat-buyer",
            "Tarek, buying from abroad",
            "An Egyptian working in the Gulf who wants a home for retirement. He cannot visit "
            "soon and worries about buying sight unseen.",
            (
                "I can't see the unit in person",
                "How do I know the paperwork is safe",
                "Exchange rates make the payments hard to plan",
            ),
            "I'm calling from Dubai. How can I trust a purchase when I can't even see the place?",
        ),
    )
}
