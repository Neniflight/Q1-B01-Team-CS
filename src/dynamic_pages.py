import mesop as me
def create_reproducible_page(header_text: str, placeholder_text: str, button_label: str, on_button_click: callable):
    with me.box(style=me.Style(padding=me.Padding.all(15), margin=me.Margin.all(15), width="100%", align_items='center', justify_content='center', flex_direction="column")):
        me.text(
            header_text,
            style=me.Style(
                font_size=24,
                font_weight="bold",
                margin=me.Margin(bottom=20)
            )
        )
        me.text(
            placeholder_text,
            style=me.Style(
                font_size=24,
                margin=me.Margin(bottom=20)
            )
        )
        me.button(
            button_label,
            on_click=on_button_click,
            color="primary",
            type="flat",
            style=me.Style(
                align_self="center",
                border=me.Border.all(me.BorderSide(width=2, color="black")),
            )
        )