# 指定要处理的文件夹根目录
$rootFolder = "C:\Users\admin\Desktop\HDR_TestSuite"

# 获取根目录下所有子文件夹
$subFolders = Get-ChildItem -Path $rootFolder -Directory

foreach ($subFolder in $subFolders) {
    # 构建 JSON 文件名，使用子文件夹名作为文件名
    $jsonFileName = Join-Path -Path $subFolder.FullName -ChildPath "metadatacr2.json"

    # 获取当前子文件夹中的所有图片文件
    $imageFiles = Get-ChildItem -Path $subFolder.FullName -Filter *.cr2

    # 遍历每个图片文件
    foreach ($imageFile in $imageFiles) {
        # 构建 exiftool 命令
        $exiftoolCommand = "exiftool -json -ExifIFD:all $($imageFile.FullName) >> $jsonFileName"

        # 执行 exiftool 命令
        Invoke-Expression -Command $exiftoolCommand
    }

    Write-Host "Exif data exported to $jsonFileName."
}