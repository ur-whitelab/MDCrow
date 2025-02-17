import pytest

from mdcrow.tools.base_tools.preprocess_tools.uniprot import (
    GetAllKnownSites,
    QueryUniprot,
)


@pytest.fixture()
def query_uniprot():
    return QueryUniprot()


def test_match_primary_accession(query_uniprot):
    mock_data = [
        {"entryType": "UniProtKB reviewed (Swiss-Prot)", "primaryAccession": "P68871"},
        {"entryType": "UniProtKB reviewed (Swiss-Prot)", "primaryAccession": "P69905"},
    ]
    assert query_uniprot._match_primary_accession(mock_data, "P69905") == [
        {
            "entryType": "UniProtKB reviewed (Swiss-Prot)",
            "primaryAccession": "P69905",
        }
    ]


def test_get_protein_name_accession(query_uniprot):
    full_names = ["Glutathione reductase", " mitochondrial"]
    short_names_included = ["Glutathione reductase", " mitochondrial", "GR", "GRase"]
    assert full_names == query_uniprot.get_protein_name(
        "gsr", "P00390", short_names=False, alternative_names=False
    )
    assert short_names_included == query_uniprot.get_protein_name(
        "gsr", "P00390", short_names=True, alternative_names=False
    )
    assert short_names_included == query_uniprot.get_protein_name(
        "gsr", "P00390", short_names=True, alternative_names=True
    )


def test_get_protein_name_no_accession(query_uniprot):
    full_names = ["Glutathione reductase", " mitochondrial"]
    short_names_included = ["Glutathione reductase", " mitochondrial", "GR", "GRase"]
    full_names_result = query_uniprot.get_protein_name(
        "gsr", short_names=False, alternative_names=False
    )
    length_full_name = 29
    length_with_short = 46
    length_with_all = 58

    assert all(name in full_names_result for name in full_names)
    assert len(full_names_result) >= length_full_name

    short_names_included_result = query_uniprot.get_protein_name(
        "gsr", short_names=True, alternative_names=False
    )
    assert all(name in short_names_included_result for name in short_names_included)
    assert len(short_names_included_result) >= length_with_short

    all_names_included_result = query_uniprot.get_protein_name(
        "gsr", short_names=True, alternative_names=True
    )
    assert all(name in all_names_included_result for name in short_names_included)
    assert len(all_names_included_result) >= length_with_all


def test_get_relevant_sites(query_uniprot):
    binding_sites = query_uniprot.get_relevant_sites("gsr", "P70619", "active")
    active_sites = query_uniprot.get_relevant_sites("gsr", "P70619", "binding")
    sites = query_uniprot.get_relevant_sites("gsr", "P70619", "sites")

    true_binding_sites = {
        "start": 413,
        "start_modifier": "EXACT",
        "end": 413,
        "end_modifier": "EXACT",
        "description": "Proton acceptor",
        "evidences": [{"evidenceCode": "ECO:0000250"}],
    }
    assert true_binding_sites in binding_sites
    assert not active_sites
    assert not sites


def test_get_all_known_sites():
    all_known_sites = GetAllKnownSites()
    site_msg = all_known_sites._run("hemoglobin", "P69905")
    assert "No known active sites." in site_msg

    assert (
        "Binding Sites: [{'start': 59, "
        "'start_modifier': 'EXACT', 'end': 59, "
        "'end_modifier': 'EXACT', 'description': "
        "'', 'evidences': [{'evidenceCode': "
        "'ECO:0000255', 'source': 'PROSITE-ProRule', "
        "'id': 'PRU00238'}]},"
    ) in site_msg

    assert (
        "Other Relevant Sites: [{'start': 9, "
        "'start_modifier': 'EXACT', 'end': 10, "
        "'end_modifier': 'EXACT', 'description': "
        "'(Microbial infection) Cleavage; by "
        "N.americanus apr-2', 'evidences': "
        "[{'evidenceCode': 'ECO:0000269', 'source': "
        "'PubMed', 'id': '12552433'}]}"
    ) in site_msg


def test_get_protein_function(query_uniprot):
    fxn_data_specific = query_uniprot.get_protein_function("hemoglobin", "P69905")
    texts = [
        "Involved in oxygen transport from the lung to the various peripheral tissues",
        (
            "Hemopressin acts as an antagonist "
            "peptide of the cannabinoid receptor "
            "CNR1 (PubMed:18077343). "
            "Hemopressin-binding efficiently blocks "
            "cannabinoid receptor CNR1 and "
            "subsequent signaling (PubMed:18077343)"
        ),
    ]
    data_texts = [comment["texts"][0]["value"] for comment in fxn_data_specific[0]]
    assert all(text in data_texts for text in texts)


def test_get_keywords(query_uniprot):
    kw = query_uniprot.get_keywords("gsr", primary_accession="P70619")
    kw_true = [
        "PTM: Disulfide bond",
        "Ligand: FAD",
        "Ligand: Flavoprotein",
        "Ligand: NADP",
        "Molecular function: Oxidoreductase",
        "Domain: Redox-active center",
        "Technical term: Reference proteome",
    ]
    assert all(k in kw for k in kw_true)

    kw_long = query_uniprot.get_keywords("gsr")
    assert len(kw_long) >= len(kw)
    assert all(k in kw_long for k in kw)


def test_get_all_sequences(query_uniprot):
    one_gfp_seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"  # noqa: E501
    all_seq = query_uniprot.get_all_sequences("gfp")
    length_all_seq = 25
    assert one_gfp_seq in all_seq
    assert len(all_seq) >= length_all_seq


def test_get_interactions(query_uniprot):
    interactions = query_uniprot.get_interactions("hemoglobin", "P69905")
    length_interactions = 13
    assert len(interactions) >= length_interactions
    i1 = [
        interactions[i]["interactantOne"]["uniProtKBAccession"]
        for i in range(len(interactions))
    ]
    assert set(i1) == {"P69905"}
    i2 = [
        interactions[i]["interactantTwo"]["uniProtKBAccession"]
        for i in range(len(interactions))
    ]
    assert all(
        i in i2
        for i in (
            "Q9NZD4",
            "Q2TAC2",
            "Q15323",
            "O76011",
            "P02042",
            "P00387",
            "P02100",
            "P29474",
            "Q6A162",
            "P0DPK4",
            "P09105",
            "P69892",
            "P68871",
        )
    )


def test_get_subunit_structure(query_uniprot):
    sus = query_uniprot.get_subunit_structure("hemoglobin", "P69905")
    sus_sus = [s["subunit structure"] for s in sus]
    assert all(
        texts in sus_sus
        for texts in [
            "Heterotetramer of two alpha chains and two beta chains in adult hemoglobin A (HbA); two alpha chains and two delta chains in adult hemoglobin A2 (HbA2); two alpha chains and two epsilon chains in early embryonic hemoglobin Gower-2; two alpha chains and two gamma chains in fetal hemoglobin F (HbF)",  # noqa: E501
            "(Microbial infection) Interacts with Staphylococcus aureus protein isdB",
        ]
    )


def test_get_sequence_info(query_uniprot):
    seq_info = query_uniprot.get_sequence_info("gsr", "P70619")
    crc64, md5 = (
        "0714FF531F90BEBA",  # pragma: allowlist secret
        "B3EF8C2F41BE8D44040346F274687F49",  # pragma: allowlist secret
    )
    sequence = "VNVGCVPKKVMWNTAVHSEFIHDHVDYGFQNCKSKFNWHVIKEKRDAYVSRLNNIYQNNLTKSHIEVIHGYATFRDGPQPTAEVNGKKFTAPHILIATGGVPTVPHENQIPGASLGITSDGFFQLEDLPSRSVIVGAGYIAVEIAGILSALGSKTSLMIRHDKVLRSFDSLISSNCTEELENAGGVEVLTVKKFSQVKEVKKTSSGLELHVVTALPGRKPTVTTIPDVDCLLWAIGRDPNSKGLNLNKLGIQTDDKGHILVDEFQNTNVKGVYAVGDVCGKALLTPVAIAAGRKLAHRLFEGKEDSRLDYDNIPTVVFSHPPIGTVGLTEDEAVHKYGKDNVKIYSTAFTPMYHAVTTRKTKCVMKMVCANKEEKVVGIHMQGIGCDEMLQGFAVAVKMGATKADFDNRVAIHPTSSEELVTLR"  # pragma: allowlist secret # noqa: E501
    length, molWeight = 424, 46301
    assert seq_info["length"] == length
    assert seq_info["molWeight"] == molWeight
    assert seq_info["crc64"] == crc64
    assert seq_info["md5"] == md5
    assert seq_info["sequence"] == sequence


def test_get_ptm_processing_info(query_uniprot):
    chains = query_uniprot.get_ptm_processing_info("hemoglobin", "P69905", "chain")
    assert {
        "start": 2,
        "start_modifier": "EXACT",
        "end": 142,
        "end_modifier": "EXACT",
        "description": "Hemoglobin subunit alpha",
        "featureId": "PRO_0000052653",
    } in chains

    assert not query_uniprot.get_ptm_processing_info(
        "hemoglobin", "P69905", "crosslink"
    )

    assert not query_uniprot.get_ptm_processing_info(
        "hemoglobin", "P69905", "disulfide-bond"
    )

    glyco = query_uniprot.get_ptm_processing_info(
        "hemoglobin", "P69905", "glycosylation"
    )
    assert all(
        g in glyco
        for g in [
            {
                "start": 8,
                "start_modifier": "EXACT",
                "end": 8,
                "end_modifier": "EXACT",
                "description": "N-linked (Glc) (glycation) lysine; alternate",
                "featureId": "",
            },
            {
                "start": 17,
                "start_modifier": "EXACT",
                "end": 17,
                "end_modifier": "EXACT",
                "description": "N-linked (Glc) (glycation) lysine; alternate",
                "featureId": "",
            },
            {
                "start": 41,
                "start_modifier": "EXACT",
                "end": 41,
                "end_modifier": "EXACT",
                "description": "N-linked (Glc) (glycation) lysine; alternate",
                "featureId": "",
            },
            {
                "start": 62,
                "start_modifier": "EXACT",
                "end": 62,
                "end_modifier": "EXACT",
                "description": "N-linked (Glc) (glycation) lysine",
                "featureId": "",
            },
        ]
    )

    i_m = query_uniprot.get_ptm_processing_info(
        "hemoglobin", "P69905", "initiator-methionine"
    )
    assert {
        "start": 1,
        "start_modifier": "EXACT",
        "end": 1,
        "end_modifier": "EXACT",
        "description": "Removed",
        "featureId": "",
    } in i_m

    assert not query_uniprot.get_ptm_processing_info(
        "hemoglobin", "P69905", "lipidation"
    )

    mr = query_uniprot.get_ptm_processing_info(
        "hemoglobin", "P69905", "modified-residue"
    )
    assert all(
        m in mr
        for m in [
            {
                "start": 4,
                "start_modifier": "EXACT",
                "end": 4,
                "end_modifier": "EXACT",
                "description": "Phosphoserine",
                "featureId": "",
            },
            {
                "start": 8,
                "start_modifier": "EXACT",
                "end": 8,
                "end_modifier": "EXACT",
                "description": "N6-succinyllysine; alternate",
                "featureId": "",
            },
            {
                "start": 9,
                "start_modifier": "EXACT",
                "end": 9,
                "end_modifier": "EXACT",
                "description": "Phosphothreonine",
                "featureId": "",
            },
            {
                "start": 12,
                "start_modifier": "EXACT",
                "end": 12,
                "end_modifier": "EXACT",
                "description": "N6-succinyllysine",
                "featureId": "",
            },
            {
                "start": 17,
                "start_modifier": "EXACT",
                "end": 17,
                "end_modifier": "EXACT",
                "description": "N6-acetyllysine; alternate",
                "featureId": "",
            },
            {
                "start": 17,
                "start_modifier": "EXACT",
                "end": 17,
                "end_modifier": "EXACT",
                "description": "N6-succinyllysine; alternate",
                "featureId": "",
            },
            {
                "start": 25,
                "start_modifier": "EXACT",
                "end": 25,
                "end_modifier": "EXACT",
                "description": "Phosphotyrosine",
                "featureId": "",
            },
        ]
    )

    pep = query_uniprot.get_ptm_processing_info("hemoglobin", "P69905", "peptide")
    assert {
        "start": 96,
        "start_modifier": "EXACT",
        "end": 104,
        "end_modifier": "EXACT",
        "description": "Hemopressin",
        "featureId": "PRO_0000455882",
    } in pep

    assert not query_uniprot.get_ptm_processing_info(
        "hemoglobin", "P69905", "propeptide"
    )

    assert not query_uniprot.get_ptm_processing_info(
        "hemoglobin", "P69905", "signal-peptide"
    )

    assert not query_uniprot.get_ptm_processing_info(
        "hemoglobin", "P69905", "transit-peptide"
    )


def test_get_3d_info(query_uniprot):
    gsr_3d = query_uniprot.get_3d_info("gsr", "P00390")
    assert all(
        i in gsr_3d
        for i in [
            {
                "database": "PDB",
                "id": "1ALG",
                "properties": [
                    {"key": "Method", "value": "NMR"},
                    {"key": "Resolution", "value": "-"},
                    {"key": "Chains", "value": "A=480-503"},
                ],
            },
            {
                "database": "PDB",
                "id": "1BWC",
                "properties": [
                    {"key": "Method", "value": "X-ray"},
                    {"key": "Resolution", "value": "2.10 A"},
                    {"key": "Chains", "value": "A=45-522"},
                ],
            },
        ]
    )


def test_get_structure_info(query_uniprot):
    beta = query_uniprot.get_structure_info("hemoglobin", "P69905", "beta")
    assert all(
        b in beta
        for b in [
            {
                "start": 45,
                "start_modifier": "EXACT",
                "end": 47,
                "end_modifier": "EXACT",
                "evidences": [
                    {"evidenceCode": "ECO:0007829", "source": "PDB", "id": "1M9P"}
                ],
            },
            {
                "start": 50,
                "start_modifier": "EXACT",
                "end": 52,
                "end_modifier": "EXACT",
                "evidences": [
                    {"evidenceCode": "ECO:0007829", "source": "PDB", "id": "6XDT"}
                ],
            },
        ]
    )

    helix = query_uniprot.get_structure_info("hemoglobin", "P69905", "helix")
    assert all(
        h in helix
        for h in [
            {
                "start": 5,
                "start_modifier": "EXACT",
                "end": 18,
                "end_modifier": "EXACT",
                "evidences": [
                    {"evidenceCode": "ECO:0007829", "source": "PDB", "id": "2W72"}
                ],
            },
            {
                "start": 19,
                "start_modifier": "EXACT",
                "end": 21,
                "end_modifier": "EXACT",
                "evidences": [
                    {"evidenceCode": "ECO:0007829", "source": "PDB", "id": "2W72"}
                ],
            },
        ]
    )

    turns = query_uniprot.get_structure_info("hemoglobin", "P69905", "turn")
    assert all(
        t in turns
        for t in [
            {
                "start": 73,
                "start_modifier": "EXACT",
                "end": 75,
                "end_modifier": "EXACT",
                "evidences": [
                    {"evidenceCode": "ECO:0007829", "source": "PDB", "id": "2W72"}
                ],
            },
            {
                "start": 91,
                "start_modifier": "EXACT",
                "end": 93,
                "end_modifier": "EXACT",
                "evidences": [
                    {"evidenceCode": "ECO:0007829", "source": "PDB", "id": "2M6Z"}
                ],
            },
        ]
    )


def test_get_ids(query_uniprot):
    hg_ids = [
        "P84792",
        "P02042",
        "P69891",
        "P69892",
        "P68871",
        "P02089",
        "P02070",
        "P02008",
        "B3EWR7",
        "P02094",
        "P01966",
        "P68872",
        "P69905",
        "P02088",
        "P02100",
        "P09105",
        "P11517",
        "P02091",
    ]
    all_ids = query_uniprot.get_ids("hemoglobin")
    print(all_ids)
    single_id = query_uniprot.get_ids("hemoglobin", single_id=True)
    assert single_id[0] in hg_ids
    assert len(single_id) == 1
    assert all(i in all_ids for i in hg_ids)


def test_get_gene_names(query_uniprot):
    specific_gene = query_uniprot.get_gene_names("gsr", "P00390")
    assert all(gene in specific_gene for gene in ["GSR", "GLUR", "GRD1"])
    all_genes = query_uniprot.get_gene_names("gsr")
    assert len(all_genes) >= len(specific_gene)
    assert all(gene in all_genes for gene in specific_gene)


def test_get_sequence_mapping(query_uniprot):
    identifiers = [
        "1A00",
        "1A01",
        "1A0U",
        "1A0Z",
        "1A3N",
        "1A3O",
        "1A9W",
        "1ABW",
        "1ABY",
        "1AJ9",
        "1B86",
        "1BAB",
        "1BBB",
        "1BIJ",
        "1BUW",
        "1BZ0",
        "1BZ1",
        "1BZZ",
        "1C7B",
        "1C7C",
        "1C7D",
        "1CLS",
        "1CMY",
        "1COH",
        "1DKE",
    ]
    mapping = query_uniprot.get_sequence_mapping("P69905")
    assert all(i in mapping for i in identifiers)


def test_get_kinetics(query_uniprot):
    with_a_t = query_uniprot.get_kinetics("rubisco", primary_accession="O85040")
    no_a_t = query_uniprot.get_kinetics("rubisco")

    assert len(with_a_t) == 1
    assert len(no_a_t) > len(with_a_t)

    assert (
        with_a_t[0][0]["kineticParameters"]["maximumVelocities"][0]["velocity"] == 2.9
    )
