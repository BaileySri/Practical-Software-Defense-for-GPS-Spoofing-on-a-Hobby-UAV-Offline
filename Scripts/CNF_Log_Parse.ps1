param ($inFile, $outCNF1, $outCNF2, $outCNF3)

#Checking parameters
while(($inFile -eq $null) -or (!(Test-Path -path $inFile))){
    if($inFile -eq $null){
        Write-Host "No input file."
    }elseif(!(Test-Path -path $inFile)){
        Write-Host "Invalid input file."
    }

    $inFile = Read-Host -Prompt "Input file name: "
    $inName = Read-Host -Prompt "Input add-on name []: "
}

if($outCNF1 -eq $null){
    if($inFile.StartsWith(".\")){
        $outCNF1 = (pwd).ToString() + "\" + (Get-Date -Format "yyyy-MM-dd")  + "-CNF1-" + $inName + ".txt"
    } else{
        $outCNF1 = (Split-Path -Path $inFile).ToString() + "\" + (Get-Date -Format "yyyy-MM-dd")  + "-CNF1-" + $inName + ".txt"
    }
}
if($outCNF2 -eq $null){
    if($inFile.StartsWith(".\")){
        $outCNF2 = (pwd).ToString() + "\" + (Get-Date -Format "yyyy-MM-dd")  + "-CNF2-" + $inName + ".txt"
    } else{
        $outCNF2 = (Split-Path -Path $inFile).ToString() + "\" + (Get-Date -Format "yyyy-MM-dd")  + "-CNF2-" + $inName + ".txt"
    }
}
if($outCNF3 -eq $null){
    if($inFile.StartsWith(".\")){
        $outCNF3 = (pwd).ToString() + "\" + (Get-Date -Format "yyyy-MM-dd")  + "-CNF3-" + $inName + ".txt"
    } else{
        $outCNF3 = (Split-Path -Path $inFile).ToString() + "\" + (Get-Date -Format "yyyy-MM-dd")  + "-CNF3-" + $inName + ".txt"
    }
}

#Outputing to file based on match
Select-String -Path $inFile -Pattern "CNF1" | select -ExpandProperty Line | ForEach-object { $_.TrimStart("CNF1,") } | ForEach-Object { $_ -creplace '.+?(?=TimeUS)', '' } | Out-File -FilePath $outCNF1 -Encoding utf8
Write-Host "Output of CNF1: $outCNF1"
Select-String -Path $inFile -Pattern "CNF2" | select -ExpandProperty Line | ForEach-object { $_.TrimStart("CNF2,") } | ForEach-Object { $_ -creplace '.+?(?=TimeUS)', '' } | Out-File -FilePath $outCNF2 -Encoding utf8
Write-Host "Output of CNF2: $outCNF2"
Select-String -Path $inFile -Pattern "CNF3" | select -ExpandProperty Line | ForEach-object { $_.TrimStart("CNF3,") } | ForEach-Object { $_ -creplace '.+?(?=TimeUS)', '' } | Out-File -FilePath $outCNF3 -Encoding utf8
Write-Host "Output of CNF3: $outCNF3"
Read-Host -Prompt "Press Enter to exit"