# Generated by Django 4.0.4 on 2022-04-18 06:26

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('aiApp', '0003_remove_student_tracks'),
    ]

    operations = [
        migrations.RenameField(
            model_name='student',
            old_name='age',
            new_name='count',
        ),
        migrations.RemoveField(
            model_name='student',
            name='description',
        ),
    ]
