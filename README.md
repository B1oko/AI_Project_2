# AI_Project_2



# Preparación de entorno

Para descargar el dataset ejecuta los siguientes comando en un "Power Shell"

```shell
# Crear carpeta 'data' si no existe
if (!(Test-Path -Path "./data")) {
    New-Item -ItemType Directory -Path "./data"
}

# Descargar el archivo ZIP usando Invoke-WebRequest (reemplazo de curl)
Invoke-WebRequest -Uri "https://www.kaggle.com/api/v1/datasets/download/ravindrasinghrana/job-description-dataset" -OutFile "./data/job-description-dataset.zip"

# Extraer el contenido del ZIP
Expand-Archive -Path "./data/job-description-dataset.zip" -DestinationPath "./data" -Force

# Confirmar finalización
Write-Host "Dataset descargado y extraído en la carpeta 'data'."

```