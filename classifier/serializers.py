from rest_framework import serializers


class ClassificationRequestSerializer(serializers.Serializer):
    abstract = serializers.CharField(required=False, allow_blank=False, max_length=20000)
    abstracts = serializers.ListField(
        child=serializers.CharField(allow_blank=False, max_length=20000),
        required=False,
        max_length=32,
    )

    def validate(self, attrs):
        if ("abstract" in attrs) == ("abstracts" in attrs):
            raise serializers.ValidationError("Provide exactly one of 'abstract' or 'abstracts'.")
        return attrs
