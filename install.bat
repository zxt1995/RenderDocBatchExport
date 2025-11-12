set batch_fbx_exporter=%APPDATA%\qrenderdoc\extensions
if not exist "%batch_fbx_exporter%" mkdir "%batch_fbx_exporter%"
xcopy "%~dp0BatchFbxExporter\*" "%batch_fbx_exporter%" /i /e /Y /C
