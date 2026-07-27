import pandas as pd
import pytest

from scripts.feature_annotation import (
    NAME_COL,
    FUNCTION_COL,
    build_feature_record,
    is_informative_metabolite_row,
    is_informative_microbiome_row,
    parse_metabolite_ids,
)


def _row(name=None, function=None):
    return pd.Series({NAME_COL: name, FUNCTION_COL: function})


def test_taxonomy_row_always_informative_even_with_na_annotation():
    feature_id = "k_Bacteria.p_Firmicutes.c_Clostridia.o_Eubacteriales.f_Lachnospiraceae.g_Blautia.s_Blautia.sp."
    assert is_informative_microbiome_row(feature_id, _row(name=None, function=None))


@pytest.mark.parametrize(
    "name,function",
    [
        ("regulator, FUR family", "Function unknown"),
        ("Uncharacterized protein", "General function prediction only"),
        (None, "Transcription"),
        ("Protein of unknown function (DUF2812)", "Function unknown"),
        ("hypothetical protein", "Not Included in Pathway or Brite; Poorly characterized; Function unknown"),
        ("some protein", None),
        ("some protein", "NA"),
    ],
)
def test_uninformative_microbiome_rows_dropped(name, function):
    assert not is_informative_microbiome_row("K06987", _row(name=name, function=function))


def test_informative_microbiome_row_kept():
    row = _row(name="lactose/L-arabinose transport system permease protein", function="Membrane transport; ABC transporters")
    assert is_informative_microbiome_row("K10189", row)


@pytest.mark.parametrize(
    "name",
    [
        "<none> Esi-0.81400025",
        "> limit Esi-0.81300193 :16",
        "C22 H38 N2 O3 Esi+8.447999",
        "C5 H11 N4 O",
        None,
        "",
    ],
)
def test_uninformative_metabolite_rows_dropped(name):
    assert not is_informative_metabolite_row(_row(name=name))


def test_informative_metabolite_row_kept():
    assert is_informative_metabolite_row(_row(name="Feruloylagmatine"))


def test_parse_metabolite_ids_extracts_present_fields():
    text = (
        "Feruloylagmatine [ C15 H22 N4 O3, tgt=,overall=96.54,db=96.53,mfg=96.5, "
        "KEGG ID=C18325,METLIN ID=7215 ]"
    )
    assert parse_metabolite_ids(text) == {"kegg": "C18325", "metlin": "7215"}


def test_parse_metabolite_ids_drops_blank_fields():
    text = "cholesta-5,7,8(14),22E-tetraen-3-one [ C27 H38 O, ..., Lipid ID=LMST01010295,METLIN ID=8389 ]"
    ids = parse_metabolite_ids(text)
    assert ids == {"lipid": "LMST01010295", "metlin": "8389"}
    assert "cas" not in ids


def test_parse_metabolite_ids_empty_when_no_ids_present():
    assert parse_metabolite_ids("[ C22 H38 N2 O3, tgt=,overall=48.75,mfg=97.4 ]") == {}


def test_build_feature_record_taxonomy_uses_feature_id_as_name():
    feature_id = "k_Bacteria.p_Firmicutes.c_Clostridia.o_Eubacteriales.f_Oscillospiraceae.g_Faecalibacterium.s_Faecalibacterium.prausnitzii"
    df = pd.DataFrame(
        {NAME_COL: [None], FUNCTION_COL: [None]}, index=pd.Index([feature_id], name="feature")
    )
    record = build_feature_record(feature_id, "microbiome", df)
    assert record == {
        "feature_id": feature_id,
        "omics_type": "microbiome",
        "annotation_type": "taxonomy",
        "name": feature_id,
        "pathway_or_function": None,
    }


def test_build_feature_record_kegg_ortholog():
    df = pd.DataFrame(
        {
            NAME_COL: ["lactose/L-arabinose transport system permease protein"],
            FUNCTION_COL: ["Membrane transport; ABC transporters"],
        },
        index=pd.Index(["K10189"], name="feature"),
    )
    record = build_feature_record("K10189", "microbiome", df)
    assert record == {
        "feature_id": "K10189",
        "omics_type": "microbiome",
        "annotation_type": "KEGG_ortholog",
        "name": "lactose/L-arabinose transport system permease protein",
        "pathway_or_function": "Membrane transport; ABC transporters",
    }


def test_build_feature_record_metabolite_includes_external_ids():
    df = pd.DataFrame(
        {
            NAME_COL: ["Feruloylagmatine"],
            FUNCTION_COL: [
                "Feruloylagmatine [ C15 H22 N4 O3, tgt=,overall=96.54,db=96.53,mfg=96.5, "
                "KEGG ID=C18325,METLIN ID=7215 ]"
            ],
        },
        index=pd.Index(["P_AQ.1388"], name="feature"),
    )
    record = build_feature_record("P_AQ.1388", "metabolite", df)
    assert record == {
        "feature_id": "P_AQ.1388",
        "omics_type": "metabolite",
        "annotation_type": "metabolite_aqueous",
        "name": "Feruloylagmatine",
        "external_ids": {"kegg": "C18325", "metlin": "7215"},
    }


def test_build_feature_record_metabolite_omits_external_ids_when_absent():
    df = pd.DataFrame(
        {
            NAME_COL: ["Lentiginosine"],
            FUNCTION_COL: ["Lentiginosine [ C8 H15 N O2, tgt=,overall=86.14 ]"],
        },
        index=pd.Index(["N_LP.999"], name="feature"),
    )
    record = build_feature_record("N_LP.999", "metabolite", df)
    assert "external_ids" not in record


def test_build_feature_record_rejects_bad_omics_type():
    df = pd.DataFrame({NAME_COL: ["x"], FUNCTION_COL: ["y"]}, index=pd.Index(["K1"], name="feature"))
    with pytest.raises(ValueError, match="omics_type"):
        build_feature_record("K1", "protein", df)
