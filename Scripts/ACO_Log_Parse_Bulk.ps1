param ($directory)

#Checking parameters
while(($directory -eq $null) -or (!(Test-Path -path $directory))){
    if($directory -eq $null){
        Write-Host "No input directory."
    }elseif(!(Test-Path -path $directory)){
        Write-Host "Invalid input directory."
    }

    $directory = Read-Host -Prompt "Input directory name: "
}

Get-ChildItem -File $directory*.log | Foreach {
$_.Name -match "(C\-.*|P\-.*)\.(log)"
$outACO1 = $directory + (Get-Date -Format "yyyy-MM-dd")  + "-ACO1-" + $matches[1] + ".txt"
$outACO2 = $directory + (Get-Date -Format "yyyy-MM-dd")  + "-ACO2-" + $matches[1] + ".txt"
Select-String -Path $_ -Pattern "ACO1" | select -ExpandProperty Line | ForEach-object { $_.TrimStart("ACO1,") } | ForEach-Object { $_ -creplace '.+?(?=TimeUS)', '' } | Out-File -FilePath $outACO1 -Encoding utf8
Write-Host "Output of ACO1: $outACO1"
Select-String -Path $_ -Pattern "ACO2" | select -ExpandProperty Line | ForEach-object { $_.TrimStart("ACO2,") } | ForEach-Object { $_ -creplace '.+?(?=TimeUS)', '' } | Out-File -FilePath $outACO2 -Encoding utf8
Write-Host "Output of ACO2: $outACO2"
}

Read-Host -Prompt "Press Enter to exit"