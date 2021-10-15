param ($inFile, $outSNS1, $outSNS2, $outSNS3, $outSNS4)

#Checking parameters
while(($inFile -eq $null) -or (!(Test-Path -path $inFile))){
    if($inFile -eq $null){
        Write-Host "No input file."
    }elseif(!(Test-Path -path $inFile)){
        Write-Host "Invalid input file."
    }

    $inFile = Read-Host -Prompt "Input file name: "
}

if($outSNS1 -eq $null){
    if($inFile.StartsWith(".\")){
        $outSNS1 = (pwd).ToString() + "\" + (Get-Date -Format "yyyy-MM-dd")  + "-SNS1.txt"
    } else{
        $outSNS1 = (Split-Path -Path $inFile).ToString() + "\" + (Get-Date -Format "yyyy-MM-dd")  + "-SNS1.txt"
    }
}
if($outSNS2 -eq $null){
    if($inFile.StartsWith(".\")){
        $outSNS2 = (pwd).ToString() + "\" + (Get-Date -Format "yyyy-MM-dd")  + "-SNS2.txt"
    } else{
        $outSNS2 = (Split-Path -Path $inFile).ToString() + "\" + (Get-Date -Format "yyyy-MM-dd")  + "-SNS2.txt"
    }
}
if($outSNS3 -eq $null){
    if($inFile.StartsWith(".\")){
        $outSNS3 = (pwd).ToString() + "\" + (Get-Date -Format "yyyy-MM-dd")  + "-SNS3.txt"
    } else{
        $outSNS3 = (Split-Path -Path $inFile).ToString() + "\" + (Get-Date -Format "yyyy-MM-dd")  + "-SNS3.txt"
    }
}
if($outSNS4 -eq $null){
    if($inFile.StartsWith(".\")){
        $outSNS4 = (pwd).ToString() + "\" + (Get-Date -Format "yyyy-MM-dd")  + "-SNS4.txt"
    } else{
        $outSNS4 = (Split-Path -Path $inFile).ToString() + "\" + (Get-Date -Format "yyyy-MM-dd")  + "-SNS4.txt"
    }
}
#Outputing to file based on match
Select-String -Path $inFile -Pattern "SNS1" | select -ExpandProperty Line | ForEach-object { $_.TrimStart("SNS1,") } | ForEach-Object { $_ -creplace '.+?(?=TimeUS)', '' } | Out-File -FilePath $outSNS1 -Encoding utf8
Write-Host "Output of SNS1: $outSNS1"
Select-String -Path $inFile -Pattern "SNS2" | select -ExpandProperty Line | ForEach-object { $_.TrimStart("SNS2,") } | ForEach-Object { $_ -creplace '.+?(?=TimeUS)', '' } | Out-File -FilePath $outSNS2 -Encoding utf8
Write-Host "Output of SNS2: $outSNS2"
Select-String -Path $inFile -Pattern "SNS3" | select -ExpandProperty Line | ForEach-object { $_.TrimStart("SNS3,") } | ForEach-Object { $_ -creplace '.+?(?=TimeUS)', '' } | Out-File -FilePath $outSNS3 -Encoding utf8
Write-Host "Output of SNS3: $outSNS3"
Select-String -Path $inFile -Pattern "SNS4" | select -ExpandProperty Line | ForEach-object { $_.TrimStart("SNS4,") } | ForEach-Object { $_ -creplace '.+?(?=TimeUS)', '' } | Out-File -FilePath $outSNS4 -Encoding utf8
Write-Host "Output of SNS4: $outSNS4"
Read-Host -Prompt "Press Enter to exit"