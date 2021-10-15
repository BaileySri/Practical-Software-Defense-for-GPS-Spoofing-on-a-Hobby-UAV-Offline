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
$outCNF1 = $directory + (Get-Date -Format "yyyy-MM-dd")  + "-CNF1-" + $matches[1] + ".txt"
$outCNF2 = $directory + (Get-Date -Format "yyyy-MM-dd")  + "-CNF2-" + $matches[1] + ".txt"
$outCNF3 = $directory + (Get-Date -Format "yyyy-MM-dd")  + "-CNF3-" + $matches[1] + ".txt"
Select-String -Path $_ -Pattern "CNF1" | select -ExpandProperty Line | ForEach-object { $_.TrimStart("CNF1,") } | ForEach-Object { $_ -creplace '.+?(?=TimeUS)', '' } | Out-File -FilePath $outCNF1 -Encoding utf8
Write-Host "Output of CNF1: $outCNF1"
Select-String -Path $_ -Pattern "CNF2" | select -ExpandProperty Line | ForEach-object { $_.TrimStart("CNF2,") } | ForEach-Object { $_ -creplace '.+?(?=TimeUS)', '' } | Out-File -FilePath $outCNF2 -Encoding utf8
Write-Host "Output of CNF2: $outCNF2"
Select-String -Path $_ -Pattern "CNF3" | select -ExpandProperty Line | ForEach-object { $_.TrimStart("CNF3,") } | ForEach-Object { $_ -creplace '.+?(?=TimeUS)', '' } | Out-File -FilePath $outCNF3 -Encoding utf8
Write-Host "Output of CNF3: $outCNF3"
}

Read-Host -Prompt "Press Enter to exit"