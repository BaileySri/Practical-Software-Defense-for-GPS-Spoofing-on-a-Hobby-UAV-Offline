param ($inFile, $outACO1, $outACO2)

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

if($outACO1 -eq $null){
    if($inFile.StartsWith(".\")){
        $outACO1 = (pwd).ToString() + "\" + (Get-Date -Format "yyyy-MM-dd")  + "-ACO1-" + $inName + ".txt"
    } else{
        $outACO1 = (Split-Path -Path $inFile).ToString() + "\" + (Get-Date -Format "yyyy-MM-dd")  + "-ACO1-" + $inName + ".txt"
    }
}
if($outACO2 -eq $null){
    if($inFile.StartsWith(".\")){
        $outACO2 = (pwd).ToString() + "\" + (Get-Date -Format "yyyy-MM-dd")  + "-ACO2-" + $inName + ".txt"
    } else{
        $outACO2 = (Split-Path -Path $inFile).ToString() + "\" + (Get-Date -Format "yyyy-MM-dd")  + "-ACO2-" + $inName + ".txt"
    }
}

#Outputing to file based on match
Select-String -Path $inFile -Pattern "ACO1" | select -ExpandProperty Line | ForEach-object { $_.TrimStart("ACO1,") } | ForEach-Object { $_ -creplace '.+?(?=TimeUS)', '' } | Out-File -FilePath $outACO1 -Encoding utf8
Write-Host "Output of ACO1: $outACO1"
Select-String -Path $inFile -Pattern "ACO2" | select -ExpandProperty Line | ForEach-object { $_.TrimStart("ACO2,") } | ForEach-Object { $_ -creplace '.+?(?=TimeUS)', '' } | Out-File -FilePath $outACO2 -Encoding utf8
Write-Host "Output of ACO2: $outACO2"
Read-Host -Prompt "Press Enter to exit"