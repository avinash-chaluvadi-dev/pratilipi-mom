from io import BytesIO
from typing import Union

from django.http import HttpResponse
from django.template.loader import get_template
from xhtml2pdf import pisa


def render_to_pdf(template_src: str, context_dict={}) -> Union[pisa.pisaDocument, None]:
    """Takes an HTML Template , dynamic context data for rendering and converts it to a PDF document"""
    template = get_template(template_src)
    html = template.render(context_dict)
    result = BytesIO()
    pdf = pisa.pisaDocument(BytesIO(html.encode("utf-8")), result)
    if not pdf.err:
        return HttpResponse(result.getvalue(), content_type="application/pdf")
    return None
