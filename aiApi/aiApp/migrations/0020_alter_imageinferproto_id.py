# Generated by Django 4.0.4 on 2022-04-19 02:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('aiApp', '0019_alter_imageinferproto_image'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imageinferproto',
            name='id',
            field=models.IntegerField(auto_created=True, primary_key=True, serialize=False, unique=True),
        ),
    ]