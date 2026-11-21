"""Write an OpenFOAM (ESI v1912, Ubuntu package `openfoam`) case for the SiPAP air path.

    python gen_case.py <case_dir> <airpath.json> <flow_L_per_min>

Steady incompressible RANS (simpleFoam, k-omega SST, wall functions), air at
~25 C (nu 1.5e-5 m2/s, rho 1.2 kg/m3). Inlet: blower outlet spigot (fixed
volumetric flow), outlet: 22 mm patient cone (p = 0). Water surface at the MAX
line is treated as a no-slip wall. Mesh: snappyHexMesh, 2 mm background,
1 mm on all walls, 0.5 mm inside the duct / SFM / dock region (16-24 cells across).
"""
import json
import os
import sys

HDR = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


def w(case, rel, cls, obj, body):
    p = os.path.join(case, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", newline="\n") as f:
        f.write(HDR.format(cls=cls, obj=obj) + body)


def main(case, info_path, q_lpm):
    info = json.load(open(info_path))
    bb = [v / 1000.0 for v in info["bbox_mm"]]
    pad = 0.003
    lo = [bb[0] - pad, bb[1] - pad, bb[2] - pad]
    hi = [bb[3] + pad, bb[4] + pad, bb[5] + pad]
    h = 0.002
    n = [max(1, round((hi[i] - lo[i]) / h)) for i in range(3)]
    loc = [v / 1000.0 + 1.3e-5 for v in info["location_in_mesh_mm"]]
    q = q_lpm / 60000.0
    probes = "\n".join("        (%.6f %.6f %.6f)" % tuple(c / 1000.0 for c in p) for p in info["probes_mm"].values())

    w(case, "system/blockMeshDict", "dictionary", "blockMeshDict", f"""
convertToMeters 1;
vertices
(
    ({lo[0]} {lo[1]} {lo[2]}) ({hi[0]} {lo[1]} {lo[2]}) ({hi[0]} {hi[1]} {lo[2]}) ({lo[0]} {hi[1]} {lo[2]})
    ({lo[0]} {lo[1]} {hi[2]}) ({hi[0]} {lo[1]} {hi[2]}) ({hi[0]} {hi[1]} {hi[2]}) ({lo[0]} {hi[1]} {hi[2]})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1) );
edges ();
boundary ( background {{ type patch; faces ( (0 3 2 1) (4 5 6 7) (0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) ); }} );
mergePatchPairs ();
""")
    w(case, "system/surfaceFeatureExtractDict", "dictionary", "surfaceFeatureExtractDict", """
airpath.stl
{
    extractionMethod    extractFromSurface;
    extractFromSurfaceCoeffs { includedAngle 150; }
    subsetFeatures { nonManifoldEdges no; openEdges yes; }
    writeObj            no;
}
""")
    w(case, "system/snappyHexMeshDict", "dictionary", "snappyHexMeshDict", f"""
castellatedMesh true;
snap            true;
addLayers       false;
geometry
{{
    airpath.stl
    {{
        type triSurfaceMesh;
        name airpath;
        regions
        {{
            inlet  {{ name inlet; }}
            outlet {{ name outlet; }}
            water  {{ name water; }}
            walls  {{ name walls; }}
        }}
    }}
    ductBox {{ type searchableBox; min (0.008 0.068 0.003); max (0.064 0.112 0.074); }}
    sfmBox  {{ type searchableBox; min (0.022 0.088 0.051); max (0.116 0.112 0.075); }}
}}
castellatedMeshControls
{{
    maxLocalCells 2000000;
    maxGlobalCells 6000000;
    minRefinementCells 0;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;
    features ( {{ file "airpath.eMesh"; level 1; }} );
    refinementSurfaces
    {{
        airpath
        {{
            level (1 1);
            regions
            {{
                inlet  {{ level (2 2); patchInfo {{ type patch; }} }}
                outlet {{ level (1 1); patchInfo {{ type patch; }} }}
                water  {{ level (1 1); patchInfo {{ type wall; }} }}
                walls  {{ level (1 1); patchInfo {{ type wall; }} }}
            }}
        }}
    }}
    resolveFeatureAngle 30;
    refinementRegions
    {{
        ductBox {{ mode inside; levels ((1E15 2)); }}
        sfmBox  {{ mode inside; levels ((1E15 2)); }}
    }}
    locationInMesh ({loc[0]} {loc[1]} {loc[2]});
    allowFreeStandingZoneFaces true;
}}
snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 50;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap true;
}}
addLayersControls
{{
    relativeSizes true; layers {{}}; expansionRatio 1.2; finalLayerThickness 0.5; minThickness 0.1;
    nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3;
    nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50;
}}
meshQualityControls
{{
    maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
    minVol 1e-16; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
    minFaceWeight 0.05; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
}}
mergeTolerance 1e-6;
""")
    w(case, "system/controlDict", "dictionary", "controlDict", f"""
application     simpleFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         2500;
deltaT          1;
writeControl    timeStep;
writeInterval   2500;
purgeWrite      1;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
// no function objects: the Ubuntu v1912 build crashes in their file output (IOstream sha1);
// post-processing is done by post.py on the ASCII fields instead.
""")
    w(case, "system/fvSchemes", "dictionary", "fvSchemes", """
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; grad(U) cellLimited Gauss linear 1; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited corrected 0.33; }
interpolationSchemes { default linear; }
snGradSchemes   { default limited corrected 0.33; }
wallDist        { method meshWave; }
""")
    w(case, "system/fvSolution", "dictionary", "fvSolution", """
solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.05; }
    "(U|k|omega)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent no;
    residualControl { p 5e-5; U 5e-6; "(k|omega)" 5e-6; }
}
relaxationFactors
{
    fields { p 0.3; }
    equations { U 0.7; k 0.6; omega 0.6; }
}
""")
    w(case, "system/decomposeParDict", "dictionary", "decomposeParDict", """
numberOfSubdomains 8;
method simple;
simpleCoeffs { n (4 2 1); delta 0.001; }
""")
    w(case, "constant/transportProperties", "dictionary", "transportProperties", """
transportModel  Newtonian;
nu              1.5e-05;
""")
    w(case, "constant/turbulenceProperties", "dictionary", "turbulenceProperties", """
simulationType RAS;
RAS { RASModel kOmegaSST; turbulence on; printCoeffs on; }
""")
    walls = "    \"(walls|water)\" { type %s; }\n"
    w(case, "0/U", "volVectorField", "U", f"""
dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{{
    inlet  {{ type flowRateInletVelocity; volumetricFlowRate constant {q:.8f}; extrapolateProfile no; value uniform (0 0 0); }}
    outlet {{ type inletOutlet; inletValue uniform (0 0 0); value uniform (0 0 0); }}
    "(walls|water)" {{ type noSlip; }}
}}
""")
    w(case, "0/p", "volScalarField", "p", """
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet  { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
""" + walls % "zeroGradient" + "}\n")
    w(case, "0/k", "volScalarField", "k", """
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0.1;
boundaryField
{
    inlet  { type turbulentIntensityKineticEnergyInlet; intensity 0.05; value uniform 0.1; }
    outlet { type inletOutlet; inletValue uniform 0.1; value uniform 0.1; }
    "(walls|water)" { type kqRWallFunction; value uniform 0.1; }
}
""")
    w(case, "0/omega", "volScalarField", "omega", """
dimensions [0 0 -1 0 0 0 0];
internalField uniform 1000;
boundaryField
{
    inlet  { type turbulentMixingLengthFrequencyInlet; mixingLength 0.0006; value uniform 1000; }
    outlet { type inletOutlet; inletValue uniform 1000; value uniform 1000; }
    "(walls|water)" { type omegaWallFunction; value uniform 1000; }
}
""")
    w(case, "0/nut", "volScalarField", "nut", """
dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet  { type calculated; value uniform 0; }
    outlet { type calculated; value uniform 0; }
    "(walls|water)" { type nutkWallFunction; value uniform 0; }
}
""")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]))
