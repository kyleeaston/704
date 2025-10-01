$TargetDir = "C:\Users\kaeas\Documents\704\template\MLSEC.competition\defender\defender\samples\malware"

# Remove extensions from all files in the directory (and subdirectories)
Get-ChildItem -Path $TargetDir -File -Recurse | ForEach-Object {
    $newName = [System.IO.Path]::GetFileNameWithoutExtension($_.Name)
    $newPath = Join-Path $_.DirectoryName $newName
    Rename-Item -Path $_.FullName -NewName $newPath
}
