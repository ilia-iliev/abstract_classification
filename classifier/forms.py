from django import forms


class ClassificationForm(forms.Form):
    abstract = forms.CharField(
        label="Abstract",
        max_length=20000,
        widget=forms.Textarea(
            attrs={
                "autofocus": True,
                "placeholder": "Paste an arXiv abstract here…",
                "rows": 12,
            }
        ),
    )
