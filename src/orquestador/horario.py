#from dagster import schedule

#@schedule(
#    cron_schedule="0 2 * * *",  # Todos los días a las 2 AM
#    job_name="full_training_pipeline",
#    execution_timezone="America/Mexico_City"
#)
#def nightly_retraining_schedule(context):
#    """
#    Programa para reentrenar automáticamente el modelo cada noche
#    """
#    return {}

#@schedule(
#    cron_schedule="0 */6 * * *",  # Cada 6 horas
#    job_name="data_processing_only",
#    execution_timezone="America/Mexico_City"
#)
#def frequent_data_processing_schedule(context):
#    """
#    Procesa nuevos datos cada 6 horas
#    """
#    return {}