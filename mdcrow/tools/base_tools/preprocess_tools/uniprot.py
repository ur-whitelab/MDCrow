import time
from enum import Enum

import requests
from langchain.tools import BaseTool
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


class SiteType(Enum):
    ACTIVE = ("ft_act_site", "active site")
    BINDING = ("ft_binding", "binding site")
    SITES = ("ft_site", "site")


class PTMType(Enum):
    CHAIN = ("ft_chain", "Chain")
    CROSSLINK = ("ft_crosslnk", "Cross-link")
    DISULFIDE_BOND = ("ft_disulfid", "Disulfide bond")
    GLYCOSYLATION = ("ft_carbohyd", "Glycosylation")
    INITIATOR_METHIONINE = ("ft_init_met", "Initiator methionine")
    LIPIDATION = ("ft_lipid", "Lipidation")
    MODIFIED_RESIDUE = ("ft_mod_res", "Modified residue")
    PEPTIDE = ("ft_peptide", "Peptide")
    PROPEPTIDE = ("ft_propep", "Propeptide")
    SIGNAL_PEPTIDE = ("ft_signal", "Signal peptide")
    TRANSIT_PEPTIDE = ("ft_transit", "Transit peptide")


class StructureMap(Enum):
    BETA = ("ft_strand", "Beta strand")
    HELIX = ("ft_helix", "Helix")
    TURN = ("ft_turn", "Turn")


class QueryUniprot:
    API_URL = "https://rest.uniprot.org"

    def get_sequence_mapping(
        self,
        query: str,
        from_db: str = "UniProtKB_AC-ID",
        to_db: str = "PDB",
        polling_interval: int = 3,
    ) -> list:
        """
        Fetch specific ID mapping from UniProt and extract the 'to' field f
        rom results.

        Args:
            query: The UniProt ID to map (e.g. 'P05067')
            from_db: The source database to map from.
                Defaults to 'UniProtKB_AC-ID'.
            to_db: The target database to map to. Defaults to 'PDB'.
            polling_interval: The interval to poll the API for results.
                Defaults to 3 seconds.

        Returns:
            A list of mapped database entries from the 'to' field if
                successful, otherwise an empty list.
        """
        with requests.Session() as session:
            session.mount(
                "https://",
                HTTPAdapter(
                    max_retries=Retry(
                        total=5,
                        backoff_factor=0.25,
                        status_forcelist=[500, 502, 503, 504],
                    )
                ),
            )
            try:
                response = session.post(
                    f"{self.API_URL}/idmapping/run",
                    data={"from": from_db, "to": to_db, "ids": query},
                )
                response.raise_for_status()
                job_id = response.json()["jobId"]

                while True:
                    response = session.get(f"{self.API_URL}/idmapping/status/{job_id}")
                    response.raise_for_status()
                    status_data = response.json()
                    if status_data.get("jobStatus") == "RUNNING":
                        print(f"Job is running. Retrying in {polling_interval}s.")
                        time.sleep(polling_interval)
                    else:
                        break

                response = session.get(f"{self.API_URL}/idmapping/details/{job_id}")
                response.raise_for_status()
                results_link = response.json().get("redirectURL")

                response = session.get(results_link)
                response.raise_for_status()
                if response.headers["Content-Type"] != "application/json":
                    raise ValueError(
                        "Expected JSON response but got a different format."
                    )

                results_json = response.json()
                results = results_json.get("results", [])
                return [r["to"] for r in results]
            except requests.HTTPError as http_err:
                print(f"HTTP error occurred: {http_err}")
                return []
            except Exception as err:
                print(f"An error occurred: {err}")
                return []

    def get_data(
        self, query: str, desired_field: str, format_type: str = "json"
    ) -> list | None:
        """
        Helper function to get data from the Uniprot API.

        Args:
            query: The query string to search (e.g. 'hemoglobin')
            desired_field: The desired field to retrieve from the API
            format_type: The format of the data to retrieve. Defaults to 'json'.

        Returns:
            The data retrieved from the API or None if no data is found.
        """
        with requests.Session() as session:
            session.mount(
                "https://",
                HTTPAdapter(
                    max_retries=Retry(
                        total=5,
                        backoff_factor=0.25,
                        status_forcelist=[500, 502, 503, 504],
                    )
                ),
            )
            url = f"https://rest.uniprot.org/uniprotkb/search?fields={desired_field}&format={format_type}&query={query}"
            try:
                response = session.get(url)
                response.raise_for_status()
                data = response.json()
            except requests.HTTPError:
                print(
                    "Requested query not found, "
                    "please try again with a valid protein identifier."
                )
                return None
            if "results" not in data or not data["results"]:
                raise ValueError(
                    "Requested query not found, "
                    "please try again with a valid protein identifier."
                )
            return data["results"]

    def _match_primary_accession(self, data: list, primary_accession: str = "") -> list:
        """
        Helper function to match the primary accession number with the data.

        Args:
            data: The data to search through
            primary_accession: The primary accession number to match

        Returns:
            The relevant data entry for the primary accession number or
                the first entry if no match is found.
        """
        if primary_accession:
            matched_data = next(
                (
                    entry
                    for entry in data
                    if entry["primaryAccession"] == primary_accession
                ),
                None,
            )
            if matched_data:
                return [matched_data]
            print(
                "The primary accession number provided does not "
                "match any entry in the data, using the first entry instead."
            )
            return [data][0]
        return [data][0]

    def get_protein_name(
        self,
        query: str,
        primary_accession: str | None = None,
        short_names: bool = True,
        alternative_names: bool = True,
    ) -> list:
        """
        Get the protein name for a specific protein, with the option to
        filter by primary accession number and to include alternative
        and shortened names.

        Args:
            query: The query string to search
            primary_accession: The primary accession number of the protein.
                Defaults to None.
            short_names: Whether to include short names in the results. Defaults
                to True.
            alternative_names: Whether to include alternative names in the
                results. Defaults to True.

        Returns:
            The protein name for the protein if found, otherwise an empty list.
                If primary_accession is provided, returns the protein name
                associated with that primary accession number, otherwise returns
                all the protein names associated with the protein.
        """
        data = self.get_data(query, desired_field="protein_name")
        if not data:
            return []
        if primary_accession:
            data = [
                entry
                for entry in data
                if entry["primaryAccession"] == primary_accession
            ]

        def _parse_names(recommended_names: dict, short_names: bool = True):
            full_name = recommended_names["fullName"]["value"].split(",")
            if not short_names:
                return full_name
            all_shortnames = recommended_names.get("shortNames", [])
            short = [name["value"] for name in all_shortnames] if all_shortnames else []
            return full_name + short

        names = []
        for d in data:
            protein_description = d["proteinDescription"]
            recommended_names = protein_description["recommendedName"]
            names.extend(_parse_names(recommended_names, short_names=short_names))
            if alternative_names:
                alt_names_data = protein_description.get("alternativeNames", [])
                names.extend(
                    _parse_names(alt_names_data[0], short_names=short_names)
                    if alt_names_data
                    else []
                )
        return names

    def _site_key(self, site_type: str) -> tuple[str, str]:
        """
        Helper function to get the desired field and associated key for
        sites (active, binding, or sites).

        Args:
            site_type: The type of site to retrieve

        Returns:
            The desired field and associated key for the type

        Raises:
            ValueError: If an invalid type is provided
        """
        try:
            site_type_map = SiteType[site_type.upper()]
        except KeyError as e:
            valid_types = ", ".join(f"'{s_type.name}'" for s_type in SiteType)
            raise ValueError(
                f"Invalid site type '{site_type}'. Valid types are: {valid_types}."
            ) from e

        return site_type_map.value

    def get_relevant_sites(
        self,
        query: str,
        primary_accession: str,
        site_type: str,
    ) -> list[dict]:
        """
        Get the relevant sites, active sites, or binding sites for a
        specific protein, given the primary accession number.

        Args:
            query: The query string to search (e.g. 'hemoglobin')
            primary_accession: The primary accession number of the protein
            site_type: The type of site to retrieve

        Returns:
            The relevant sites for the protein with the given primary accession number
                The list contains a dict for each site with the following keys:
                - 'start': The start position of the site
                - 'start_modifier': The start position modifier of the site
                - 'end': The end position of the site
                - 'end_modifier': The end position modifier of the site
                - 'description': The description of the site
                - 'evidences': The evidences for the site
        """
        desired_field, associated_key = self._site_key(site_type)
        if not desired_field:
            return []
        data = self.get_data(query, desired_field=desired_field)
        if not data:
            return []
        data = self._match_primary_accession(data, primary_accession)
        all_sites = {}
        features = [
            feature
            for feature in data[0]["features"]
            if feature["type"].lower() == associated_key
        ]
        all_sites[primary_accession] = features
        if not all_sites:
            return []
        relevant_sites = all_sites.get(primary_accession)
        if not relevant_sites:
            return []

        sites = []
        for site in relevant_sites:
            start = site["location"]["start"]["value"]
            start_modifier = site["location"]["start"].get("modifier", "")
            end = site["location"]["end"]["value"]
            end_modifier = site["location"]["end"].get("modifier", "")
            description = site["description"]
            evidences = site.get("evidences", [])
            sites.append(
                {
                    "start": start,
                    "start_modifier": start_modifier,
                    "end": end,
                    "end_modifier": end_modifier,
                    "description": description,
                    "evidences": evidences,
                }
            )
        return sites

    def get_protein_function(
        self, query: str, primary_accession: str | None = None
    ) -> list:
        """
        Get the protein function for a specific protein, with the option to
        filter by primary accession number.

        Args:
            query: The query string to search (e.g. 'hemoglobin')
            primary_accession: The primary accession number of the protein.
                Defaults to None.

        Returns:
            The protein function for the protein.
                If primary_accession is provided, returns the protein function
                associated with that primary accession number, otherwise returns
                all the protein functions associated with the protein.
        """
        data = self.get_data(query, desired_field="cc_function")
        if not data:
            return []
        if primary_accession:
            data = self._match_primary_accession(data, primary_accession)
        return [
            entry["comments"]
            for entry in data
            if "commentType" not in entry["comments"]
        ]

    def get_keywords(self, query: str, primary_accession: str | None = None) -> list:
        """
        Get the keywords for a specific protein, with the option to filter by
        primary accession number.

        Args:
            query: The query string to search (e.g. 'hemoglobin')
            primary_accession: The primary accession number of the protein.
                Defaults to None.

        Returns:
            The keywords for the protein.
                If primary_accession is provided, returns the keywords
                associated with that primary accession number. Otherwise,
                returns all the keywords associated with the protein
        """
        keywords = self.get_data(query, desired_field="keyword")
        if not keywords:
            return []
        if primary_accession:
            keywords = self._match_primary_accession(keywords, primary_accession)
            return [
                f"{entry['category']}: {entry['name']}"
                for entry in keywords[0]["keywords"]
            ]
        return [
            f"{entry['category']}: {entry['name']}"
            for kw_row in keywords
            for entry in kw_row["keywords"]
        ]

    def get_all_sequences(self, query: str) -> list:
        """
        Get all the sequences for a specific protein.

        Args:
            query: The query string to search (e.g. 'hemoglobin')

        Returns:
            The sequences for the protein
        """
        data = self.get_data(query, desired_field="sequence")
        return [entry["sequence"]["value"] for entry in data] if data else []

    def get_interactions(self, query: str, primary_accession: str) -> list:
        """
        Get the interactions for a specific protein, given the primary accession
        number.

        Args:
            query: The query string to search (e.g. 'hemoglobin')
            primary_accession: The primary accession number of the protein
                (required)

        Returns:
            The interactions for the protein with the given primary accession
                number
        """
        data = self.get_data(query, desired_field="cc_interaction")
        if not data:
            return []
        data = self._match_primary_accession(data, primary_accession)
        return next(
            comment["interactions"]
            for interaction in data
            for comment in interaction["comments"]
        )

    def get_subunit_structure(self, query: str, primary_accession: str) -> list:
        """
        Get the subunit structure information for a specific protein, given the
        primary accession number.

        Args:
            query: The query string to search (e.g. 'hemoglobin')
            primary_accession: The primary accession number of the protein

        Returns:
            The subunit structure information for the protein with the given
                primary accession number, along with the evidence
        """
        data = self.get_data(query, desired_field="cc_subunit")
        if not data:
            return []
        data = self._match_primary_accession(data, primary_accession)
        texts = [comment["texts"] for comment in data[0]["comments"]]
        if not texts:
            print("No subunit structure information found.")
            return []
        return [
            {
                "subunit structure": text["value"],
                "evidence": text.get("evidences", "No evidence provided"),
            }
            for text_list in texts
            for text in text_list
        ]

    def get_sequence_info(self, query: str, primary_accession: str) -> dict:
        """
        Get the sequence information for a specific protein, given the primary
        accession number.

        Args:
            query: The query string to search (e.g. 'hemoglobin')
            primary_accession: The primary accession number of the protein

        Returns:
            The sequence information for the protein with the given accession
                The dictionary contains the following keys:
                - 'sequence': The sequence of the protein
                - 'length': The length of the protein sequence
                - 'molWeight': The molecular weight of the protein
                - 'crc64': The CRC64 hash of the protein sequence (probably not useful)
                - 'md5': The MD5 hash of the protein sequence (probably not useful)
        """
        seq_info = self.get_data(query, desired_field="sequence")
        if not seq_info:
            return {}
        seq_info_specific = self._match_primary_accession(seq_info, primary_accession)[
            0
        ]["sequence"]
        seq_info_specific["sequence"] = seq_info_specific.pop("value")
        return seq_info_specific

    def _ptm_key(self, ptm_key: str) -> tuple[str, str]:
        """
        Helper function to get the desired field and associated key for PTM/
        Processing (e.g., chain, crosslink, disulfide-bond, etc.).

        Args:
            ptm_key: The PTM/Processing key to retrieve.

        Returns:
            The desired field and associated key for the PTM/Processing key.

        Raises:
            ValueError: If an invalid PTM/Processing key is provided.
        """
        normalized_key = ptm_key.replace(" ", "_").replace("-", "_").lower()
        try:
            ptm_type = PTMType[normalized_key.upper()]
        except KeyError as e:
            valid_keys = ", ".join(
                f"'{key.name.replace('_', ' ').lower()}'" for key in PTMType
            )
            raise ValueError(
                "Invalid PTM/Processing key, "
                f"please use one of the following: {valid_keys}."
            ) from e
        return ptm_type.value

    def get_ptm_processing_info(
        self,
        query: str,
        primary_accession: str,
        ptm_key: str,
    ) -> list[dict]:
        """
        Get the ptm/processing information for a specific protein, given the
        primary accession number.

        Args:
            query: The query string to search (e.g. 'hemoglobin')
            primary_accession: The primary accession number of the protein
            ptm_key: The PTM/Processing key to retrieve

        Returns:
            The relevant information for the protein with the given primary
                accession number
                The list contains a dictionary for each object with the
                following keys:
                - 'start': The start position
                - 'start_modifier': The start position modifier
                - 'end': The end position
                - 'end_modifier': The end position modifier
                - 'description': The description
                - 'featureId': The feature ID
        """
        desired_field, associated_key = self._ptm_key(ptm_key)
        if not desired_field:
            return []
        data = self.get_data(query, desired_field=desired_field)
        if not data:
            return []
        data = self._match_primary_accession(data, primary_accession)

        structure_info = []
        relevant_fields = [
            feature
            for feature in data[0]["features"]
            if feature["type"] == associated_key
        ]
        for field in relevant_fields:
            start_ = field["location"]["start"]["value"]
            start_modifier = field["location"]["start"].get("modifier", "")
            end_ = field["location"]["end"]["value"]
            end_modifier = field["location"]["end"].get("modifier", "")
            description = field.get("description", "")
            featureid = field.get("featureId", "")
            structure_info.append(
                {
                    "start": start_,
                    "start_modifier": start_modifier,
                    "end": end_,
                    "end_modifier": end_modifier,
                    "description": description,
                    "featureId": featureid,
                }
            )
        return structure_info

    def _structure_key(self, structure_key: str) -> tuple[str, str]:
        """
        Helper function to get the desired field and associated key for
        structure beta, helix, turn).

        Args:
            structure_key: The structure key to retrieve

        Returns:
            The desired field and associated key for the structure key

        Raises:
            ValueError: If an invalid structure key is provided
        """
        try:
            structure_key_map = StructureMap[structure_key.upper()]
        except KeyError as e:
            valid_keys = ", ".join(f"'{key.name}'" for key in StructureMap)
            raise ValueError(
                f"Invalid structure key '{structure_key}'. "
                f"Valid keys are: {valid_keys}."
            ) from e
        return structure_key_map.value

    def get_3d_info(self, query: str, primary_accession: str) -> list:
        """
        Get the 3D structure information for a specific protein, given the
        primary accession number.

        Args:
            query: The query string to search (e.g. 'hemoglobin')
            primary_accession: The primary accession number of the protein

        Returns:
            The 3D structure information for the protein with the given primary
                accession number
        """
        data = self.get_data(query, desired_field="structure_3d")
        if not data:
            return []
        data = self._match_primary_accession(data, primary_accession)
        return data[0]["uniProtKBCrossReferences"]

    def get_structure_info(
        self,
        query: str,
        primary_accession: str,
        structure_key: str,
    ) -> list[dict]:
        """
        Get the structure information for a specific protein, given the primary
        accession number, including either beta sheets, helices, or turns.

        Args:
            query: The query string to search (e.g. 'hemoglobin')
            primary_accession: The primary accession number of the protein
            structure_key: The structure key to retrieve

        Returns:
            The structure information for the protein with the given primary
                accession number
                The list contains a dictionary for each structure with the
                following keys:
                - 'start': The start position
                - 'start_modifier': The start position modifier
                - 'end': The end position
                - 'end_modifier': The end position modifier
                - 'evidences': The evidences for the structure
        """
        desired_field, associated_key = self._structure_key(structure_key)
        if not desired_field:
            return []
        data = self.get_data(query, desired_field=desired_field)
        if not data:
            return []
        data = self._match_primary_accession(data, primary_accession)

        structure_info = []
        relevant_fields = [
            feature
            for feature in data[0]["features"]
            if feature["type"] == associated_key
        ]

        for field in relevant_fields:
            start_ = field["location"]["start"]["value"]
            start_modifier = field["location"]["start"].get("modifier", "")
            end_ = field["location"]["end"]["value"]
            end_modifier = field["location"]["end"].get("modifier", "")
            evidences = field.get("evidences", [])
            structure_info.append(
                {
                    "start": start_,
                    "start_modifier": start_modifier,
                    "end": end_,
                    "end_modifier": end_modifier,
                    "evidences": evidences,
                }
            )
        return structure_info

    def get_ids(
        self, query: str, single_id: bool = False, include_uniprotkbids=False
    ) -> list:
        """
        Get the IDs for a specific protein.

        Args:
            query: The query string to search (e.g. 'hemoglobin')
            single_id: Whether to return a single ID or all IDs. Defaults to
                False.
            include_uniprotkbids: Whether to include UniProtKB IDs in the
                results. Defaults to False.

        Returns:
            The IDs for the protein
        """
        ids_ = self.get_data(query, desired_field="id")
        all_ids = [entry["primaryAccession"] for entry in ids_] if ids_ else []
        if include_uniprotkbids:
            all_ids + [entry["uniProtkbId"] for entry in ids_] if ids_ else []
        accession = self.get_data(query, desired_field="accession")
        (
            all_ids + [entry["primaryAccession"] for entry in accession]
            if accession
            else []
        )
        if single_id:
            return [all_ids[0]] if all_ids else []
        return list(set(all_ids))

    def get_gene_names(self, query: str, primary_accession: str | None = None) -> list:
        """
        Get the gene names for a specific protein, with the option to filter by
        primary accession number.

        Args:
            query: The query string to search (e.g. 'hemoglobin')
            primary_accession: The primary accession number of the protein.
                Defaults to None.

        Returns:
            The gene names for the protein if gene names are found, otherwise an
                empty list.
                If primary_accession is provided, returns the gene names
                associated with that primary accession number, otherwise returns
                all the gene names associated with the protein.
        """
        data = self.get_data(query, desired_field="gene_names")
        if not data:
            return []
        if primary_accession:
            data = [
                entry
                for entry in data
                if entry["primaryAccession"] == primary_accession
            ]
        all_genes = []
        for i in range(len(data)):
            if "genes" not in data[i]:
                continue
            gene_info = data[i]["genes"]
            gene_name = [gene_name["geneName"]["value"] for gene_name in gene_info]
            synonyms = [
                value["value"]
                for synonym in gene_info
                if "synonyms" in synonym
                for value in synonym["synonyms"]
            ]
            orfNames = [
                value["value"]
                for orf in gene_info
                if "orfNames" in orf
                for value in orf["orfNames"]
            ]
            orderedlocus = [
                value["value"]
                for ordered in gene_info
                if "orderedLocusNames" in ordered
                for value in ordered["orderedLocusNames"]
            ]
            all_genes.extend(gene_name + synonyms + orfNames + orderedlocus)
        return all_genes

    def get_kinetics(self, query: str, primary_accession: str | None = None) -> list:
        """
        Get the kinetics information for a specific protein, given the primary
        accession number.

        Args:
            query: The query string to search (e.g. 'hemoglobin')
            primary_accession: The primary accession number of the protein

        Returns:
            The kinetics information for the protein with the given primary
                accession number
        """
        data = self.get_data(query, desired_field="kinetics")
        if not data:
            return []

        if primary_accession:
            data = self._match_primary_accession(data, primary_accession)

        return [entry["comments"] for entry in data if entry["comments"]]


class MapProteinRepresentation(BaseTool):
    name = "MapProteinRepresentation"
    description = (
        "Fetch specific ID mapping from UniProt. "
        "You must specify the database to map from and to, "
        "as well as the representation of the protein. "
        "The defaults are 'UniProtKB_AC-ID' and 'PDB', respectively."
    )
    uniprot = QueryUniprot()

    def _run(
        self, query: str, src_db: str = "UniProtKB_AC-ID", dst_db: str = "PDB"
    ) -> str:
        """use the tool."""
        try:
            mapped_ids = self.uniprot.get_sequence_mapping(
                query, from_db=src_db, to_db=dst_db
            )
            return str(mapped_ids)
        except Exception as e:
            return str(e)

    async def _arun(self, query: str, src_db: str | None, dst_db: str | None) -> str:
        """use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution.")


class UniprotID2Name(BaseTool):
    name = "UniprotID2Name"
    description = (
        "Get the protein name for a specific protein, "
        "with the option to filter by primary accession"
        "number. If you have the primary accession "
        "number, you can use it to filter the results. "
        "Otherwise, all names associated with the "
        "protein will be returned. Input the uniprot ID"
        "of the protein."
    )
    uniprot = QueryUniprot()
    all_names: bool = True

    def __init__(self, all_names: bool = True):
        super().__init__()
        self.all_names = all_names

    def _run(self, query: str, primary_accession: str = "") -> str:
        """use the tool."""
        try:
            names = self.uniprot.get_protein_name(
                query,
                primary_accession=primary_accession,
                short_names=self.all_names,
                alternative_names=self.all_names,
            )
            return ", ".join(names)
        except Exception as e:
            return str(e)

    async def _arun(self, query: str, primary_accession: str) -> str:
        """use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution.")


class GetBindingSites(BaseTool):
    name = "GetBindingSites"
    description = (
        "Get the binding sites known for a specific "
        "protein, given the primary accession number. "
        "Both the query string and primary accession "
        "number are required. "
    )
    uniprot = QueryUniprot()

    def _run(self, query: str, primary_accession: str = "") -> str:
        """use the tool."""
        try:
            sites = self.uniprot.get_relevant_sites(query, primary_accession, "binding")
            return str(sites)
        except Exception as e:
            return str(e)

    async def _arun(self, query: str, primary_accession: str) -> str:
        """use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution.")


class GetActiveSites(BaseTool):
    name = "GetActiveSites"
    description = (
        "Get the active sites known for a specific "
        "protein, given the primary accession number. "
        "Both the query string and primary accession "
        "number are required. "
    )
    uniprot = QueryUniprot()

    def _run(self, query: str, primary_accession: str = "") -> str:
        """use the tool."""
        try:
            sites = self.uniprot.get_relevant_sites(query, primary_accession, "active")
            return str(sites)
        except Exception as e:
            return str(e)

    async def _arun(self, query: str, primary_accession: str) -> str:
        """use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution.")


class GetRelevantSites(BaseTool):
    name = "GetRelevantSites"
    description = (
        "Get the relevant sites for a specific protein, "
        "given the primary accession number. You must "
        "provide the query string and primary accession "
        "number. The relevant sites are sites that are "
        "known to be important for the protein's function, "
        "but are not necessarily active or binding sites."
    )
    uniprot = QueryUniprot()

    def _run(self, query: str, primary_accession: str = "") -> str:
        """use the tool."""
        try:
            sites = self.uniprot.get_relevant_sites(query, primary_accession, "sites")
            return str(sites)
        except Exception as e:
            return str(e)

    async def _arun(self, query: str, primary_accession: str) -> str:
        """use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution.")


class GetAllKnownSites(BaseTool):
    name = "GetAllKnownSites"
    description = (
        "Get all known sites for a specific protein, "
        "given the primary accession number. You must "
        "provide the query string and primary accession "
        "number. This tool is a one-stop shop to get all known sites "
        "for the protein, including active sites, binding "
        "sites, and other relevant sites."
    )
    uniprot = QueryUniprot()

    def _run(self, query: str, primary_accession: str = "") -> str:
        """use the tool."""
        try:
            active_sites = self.uniprot.get_relevant_sites(
                query, primary_accession, "active"
            )
            active_sites_msg = (
                f"Active Sites: {active_sites}"
                if active_sites
                else "No known active sites."
            )

            binding_sites = self.uniprot.get_relevant_sites(
                query, primary_accession, "binding"
            )
            binding_sites_msg = (
                f"Binding Sites: {binding_sites}"
                if binding_sites
                else "No known binding sites."
            )

            sites = self.uniprot.get_relevant_sites(query, primary_accession, "sites")
            sites_msg = (
                f"Other Relevant Sites: {sites}"
                if sites
                else "No other relevant sites."
            )

            return active_sites_msg + "\n" + binding_sites_msg + "\n" + sites_msg
        except Exception as e:
            return str(e)

    async def _arun(self, query: str, primary_accession: str) -> str:
        """use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution.")


class GetProteinFunction(BaseTool):
    name = "GetProteinFunction"
    description = (
        "Get the protein function for a specific protein, "
        "with the option to filter by primary accession number. "
        "If you have the primary accession number, you can use "
        "it to filter the results. Otherwise, all functions "
        "associated with the protein will be returned. "
        "Input the uniprot ID of the protein."
    )
    uniprot = QueryUniprot()

    def _run(self, query: str, primary_accession: str = "") -> str:
        """use the tool."""
        try:
            functions = self.uniprot.get_protein_function(
                query, primary_accession=primary_accession
            )
            return ", ".join(functions)
        except Exception as e:
            return str(e)

    async def _arun(self, query: str, primary_accession: str) -> str:
        """use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution.")


class GetProteinAssociatedKeywords(BaseTool):
    name = "GetProteinAssociatedKeywords"
    description = (
        "Get the keywords associated with a specific protein, with "
        "the option to filter by primary accession number. If you "
        "have the primary accession number, you can use it to "
        "filter the results. Otherwise, all keywords associated "
        "with the protein will be returned. Input the uniprot ID "
        "of the protein."
    )
    uniprot = QueryUniprot()

    def _run(self, query: str, primary_accession: str = "") -> str:
        """use the tool."""
        try:
            keywords = self.uniprot.get_keywords(
                query, primary_accession=primary_accession
            )
            return ", ".join(keywords)
        except Exception as e:
            return str(e)

    async def _arun(self, query: str, primary_accession: str) -> str:
        """use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution.")


class GetAllSequences(BaseTool):
    name = "GetAllSequences"
    description = (
        "Get all the sequences for a specific protein. "
        "Input the uniprot ID of the protein."
        "This tool will return all sequences associated with the protein."
    )
    uniprot = QueryUniprot()

    def _run(self, query: str) -> str:
        """use the tool."""
        try:
            sequences = self.uniprot.get_all_sequences(query)
            return ", ".join(sequences)
        except Exception as e:
            return str(e)

    async def _arun(self, query: str) -> str:
        """use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution.")


class GetInteractions(BaseTool):
    name = "GetInteractions"
    description = (
        "Get the interactions for a specific protein, given the "
        "primary accession number. Both the query string and primary "
        "accession number are required. This tool will return the "
        "interactions for the protein."
    )
    uniprot = QueryUniprot()

    def _run(self, query: str, primary_accession: str = "") -> str:
        """use the tool."""
        try:
            interactions = self.uniprot.get_interactions(query, primary_accession)
            return str(interactions)
        except Exception as e:
            return str(e)

    async def _arun(self, query: str, primary_accession: str) -> str:
        """use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution.")


class GetSubunitStructure(BaseTool):
    name = "GetSubunitStructure"
    description = (
        "Get the subunit structure information for a specific protein, "
        "given the primary accession number. Both the query string and "
        "primary accession number are required. This tool will return "
        "the subunit structure information for the protein."
    )
    uniprot = QueryUniprot()

    def _run(self, query: str, primary_accession: str = "") -> str:
        """use the tool."""
        try:
            structure_info = self.uniprot.get_subunit_structure(
                query, primary_accession
            )
            return str(structure_info)
        except Exception as e:
            return str(e)

    async def _arun(self, query: str, primary_accession: str) -> str:
        """use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution.")


class GetSequenceInfo(BaseTool):
    name = "GetSequenceInfo"
    description = (
        "Get the sequence information for a specific protein, "
        "given the primary accession number. Both the query string "
        "and primary accession number are required. This tool will "
        "return the sequence, length, and molecular weight. "
    )
    uniprot = QueryUniprot()

    def _run(self, query: str, primary_accession: str = "") -> str:
        """use the tool."""
        try:
            sequence_info = self.uniprot.get_sequence_info(query, primary_accession)
            # remove crc64 and md5 keys, as they are not useful to the agent
            sequence_info.pop("crc64", None)
            sequence_info.pop("md5", None)
            return str(sequence_info)
        except Exception as e:
            return str(e)

    async def _arun(self, query: str, primary_accession: str) -> str:
        """use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution.")


class GetPDBProcessingInfo(BaseTool):
    name = "GetPDBProcessingInfo"
    description = (
        "Get the processing information for a specific protein, "
        "given the primary accession number. Both the query string "
        "and primary accession number are required. Input the query, accession "
        "number, and the type of processing information to retrieve (e.g., "
        "chain, crosslink, disulfide-bond, etc.). Here is a list of the "
        "processing types you can retrieve: chain, crosslink, disulfide-bond, "
        "glycosylation, initiator-methionine, lipidation, modified-residue, "
        "peptide, propeptide, signal-peptide, transit-peptide"
    )
    uniprot = QueryUniprot()

    def _run(
        self, query: str, processing_type: str, primary_accession: str = ""
    ) -> str:
        """use the tool."""
        try:
            processing_info = self.uniprot.get_ptm_processing_info(
                query, primary_accession, processing_type
            )
            return str(processing_info)
        except Exception as e:
            return str(e)

    async def _arun(self, query: str, primary_accession: str) -> str:
        """use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution.")


class GetPDB3DInfo(BaseTool):
    name = "GetPDB3DInfo"
    description = (
        "Get the 3D structure information for a specific protein, "
        "given the primary accession number. Both the query string "
        "and primary accession number are required. This tool will "
        "return information from the PDB database for the protein, "
        "including the PDB ID, chain, and resolution."
    )
    uniprot = QueryUniprot()

    def _run(self, query: str, primary_accession: str = "") -> str:
        """use the tool."""
        try:
            structure_info = self.uniprot.get_3d_info(query, primary_accession)
            return str(structure_info)
        except Exception as e:
            return str(e)

    async def _arun(self, query: str, primary_accession: str) -> str:
        """use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution.")


class GetTurnsBetaSheetsHelices(BaseTool):
    name = "GetTurnsBetaSheetsHelices"
    description = (
        "Get the number and location of turns, beta sheets, and helices "
        "for a specific protein, given the primary accession number. Both "
        "the query string and primary accession number are required. This "
        "tool will return the number and location of turns, beta sheets, and "
        "helices for the protein. "
    )
    uniprot = QueryUniprot()

    def _run(self, query: str, primary_accession: str = "") -> str:
        """use the tool."""
        try:
            turns = self.uniprot.get_structure_info(query, primary_accession, "turn")
            beta_sheets = self.uniprot.get_structure_info(
                query, primary_accession, "beta"
            )
            helices = self.uniprot.get_structure_info(query, primary_accession, "helix")
            return f"Turns: {turns}\nBeta sheets: {beta_sheets}\nHelices: {helices}"
        except Exception as e:
            return str(e)

    async def _arun(self, query: str, primary_accession: str) -> str:
        """use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution.")


class GetUniprotID(BaseTool):
    name = "GetUniprotID"
    description = (
        "Get the UniProt ID for a specific protein. "
        "Input the query string of the protein. "
        "This tool will return the UniProt ID of the protein. "
        "You can optionally specify whether you want to return "
        "all IDs or just one ID. By default, a single ID will be "
        "returned."
    )
    uniprot = QueryUniprot()
    include_uniprotkbids: bool = False

    def __init__(self, include_uniprotkbids: bool = False):
        super().__init__()
        self.include_uniprotkbids = include_uniprotkbids

    def _run(self, query: str, all_ids: bool = False) -> str:
        """use the tool."""
        try:
            ids = self.uniprot.get_ids(
                query,
                single_id=not all_ids,
                include_uniprotkbids=self.include_uniprotkbids,
            )
            return ", ".join(ids)
        except Exception as e:
            return str(e)

    async def _arun(self, query: str, all_ids: bool) -> str:
        """use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution.")


class GetGeneNames(BaseTool):
    name = "GetGeneNames"
    description = (
        "Get the gene names associated with a specific protein, "
        "with the option to filter by primary accession number. "
        "If you have the primary accession number, you can use it "
        "to filter the results. Otherwise, all gene names associated "
        "with the protein will be returned. Input the uniprot ID of "
        "the protein."
    )
    uniprot = QueryUniprot()

    def _run(self, query: str, primary_accession: str = "") -> str:
        """use the tool."""
        try:
            gene_names = self.uniprot.get_gene_names(
                query, primary_accession=primary_accession
            )
            return ", ".join(gene_names)
        except Exception as e:
            return str(e)

    async def _arun(self, query: str, primary_accession: str) -> str:
        """use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution.")


class GetKineticProperties(BaseTool):
    name = "GetKineticProperties"
    description = (
        "Get the kinetics information for a specific protein, "
        "given the primary accession number. "
        "Both the query string and primary accession number are required. "
    )
    uniprot = QueryUniprot()

    def _run(self, query: str, primary_accession: str = "") -> str:
        """use the tool."""
        try:
            kinetics = self.uniprot.get_kinetics(query, primary_accession)
            return str(kinetics)
        except Exception as e:
            return str(e)

    async def _arun(self, query: str, dependency: str, primary_accession: str) -> str:
        """use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution.")
