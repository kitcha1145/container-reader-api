# Generated by Django 4.0.4 on 2022-04-18 07:17

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('aiApp', '0005_student_age'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Student',
            new_name='UserManagement',
        ),
    ]