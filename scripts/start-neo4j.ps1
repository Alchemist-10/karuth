<#
.SYNOPSIS
    Launcher for the Neo4j 5.26 Community instance used by the hackp project.
    Run it from any directory - it switches into the Neo4j bin folder itself.

.EXAMPLE
    ./scripts/start-neo4j.ps1            # start console (foreground; Ctrl+C to stop)
    ./scripts/start-neo4j.ps1 console
    ./scripts/start-neo4j.ps1 start      # run in background
    ./scripts/start-neo4j.ps1 stop
    ./scripts/start-neo4j.ps1 status
    ./scripts/start-neo4j.ps1 restart
#>

param(
    [string]$SubCommand = 'console',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = 'Stop'

# --- Config: point at your Neo4j install ---
$Neo4jHome = 'C:\Users\Akshay VS\karuth-neo4j\neo4j-community-5.26.0'
$Neo4jBat  = Join-Path $Neo4jHome 'bin\neo4j.bat'

if (-not (Test-Path $Neo4jBat)) {
    Write-Error "Neo4j not found at '$Neo4jBat'. Fix `$Neo4jHome at the top of this script."
    exit 1
}

# --- Java: Neo4j 5 requires a JDK 17+ ---
if (-not $env:JAVA_HOME) {
    $javaCmd = Get-Command java -ErrorAction SilentlyContinue
    if ($javaCmd) {
        # ...\jdk-xx\bin\java.exe  ->  JAVA_HOME is the grandparent of java.exe
        $env:JAVA_HOME = Split-Path -Parent (Split-Path -Parent $javaCmd.Source)
        Write-Host "JAVA_HOME not set; deriving it from 'java': $env:JAVA_HOME" -ForegroundColor Yellow
    }
    else {
        Write-Error 'Java not found. Install a JDK 17+ and set JAVA_HOME before running Neo4j.'
        exit 1
    }
}
Write-Host "JAVA_HOME = $env:JAVA_HOME" -ForegroundColor DarkGray

# --- Run ---
$binDir = Split-Path -Parent $Neo4jBat
Write-Host "neo4j.bat $SubCommand $($Rest -join ' ')  (cwd: $binDir)" -ForegroundColor Cyan
Push-Location $binDir
try {
    & $Neo4jBat $SubCommand @Rest
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
