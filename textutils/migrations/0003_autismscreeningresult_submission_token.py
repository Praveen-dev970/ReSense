import uuid
from django.db import migrations, models


def populate_submission_tokens(apps, schema_editor):
    ScreeningResult = apps.get_model(
        'textutils',
        'AutismScreeningResult'
    )

    for record in ScreeningResult.objects.all().iterator():
        record.submission_token = uuid.uuid4()
        record.save(update_fields=['submission_token'])


class Migration(migrations.Migration):

    dependencies = [
        ('textutils', '0002_autismscreeningresult_age_months'),
    ]

    operations = [
        migrations.AddField(
            model_name='autismscreeningresult',
            name='submission_token',
            field=models.UUIDField(
                default=uuid.uuid4,
                editable=False
            ),
        ),
        migrations.RunPython(
            populate_submission_tokens,
            migrations.RunPython.noop
        ),
        migrations.AlterField(
            model_name='autismscreeningresult',
            name='submission_token',
            field=models.UUIDField(
                default=uuid.uuid4,
                editable=False,
                unique=True
            ),
        ),
    ]