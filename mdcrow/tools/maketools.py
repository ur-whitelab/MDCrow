import os

import numpy as np
from dotenv import load_dotenv
from langchain import agents
from langchain.base_language import BaseLanguageModel
from langchain_openai import OpenAIEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mdcrow.utils import PathRegistry

from .base_tools import (
    CleaningToolFunction,
    ComputeAcylindricity,
    ComputeAngles,
    ComputeAsphericity,
    ComputeDSSP,
    ComputeGyrationTensor,
    ComputeLPRMSD,
    ComputeRelativeShapeAntisotropy,
    ComputeRMSD,
    ComputeRMSF,
    ContactsTool,
    DistanceMatrixTool,
    GetActiveSites,
    GetAllKnownSites,
    GetAllSequences,
    GetBindingSites,
    GetGeneNames,
    GetInteractions,
    GetKineticProperties,
    GetPDB3DInfo,
    GetPDBProcessingInfo,
    GetProteinAssociatedKeywords,
    GetProteinFunction,
    GetRelevantSites,
    GetSequenceInfo,
    GetSubunitStructure,
    GetTurnsBetaSheetsHelices,
    GetUniprotID,
    HydrogenBondTool,
    ListRegistryPaths,
    MapProteinRepresentation,
    ModifyBaseSimulationScriptTool,
    MomentOfInertia,
    PackMolTool,
    PCATool,
    PPIDistance,
    ProteinName2PDBTool,
    RadiusofGyrationTool,
    RDFTool,
    Scholar2ResultLLM,
    SetUpandRunFunction,
    SimulationOutputFigures,
    SmallMolPDB,
    SolventAccessibleSurfaceArea,
    SummarizeProteinStructure,
    UniprotID2Name,
    VisualizeProtein,
)


def make_all_tools(
    llm: BaseLanguageModel,
    human=False,
    modifysim_no_run=False,
):
    load_dotenv()
    all_tools = []
    path_instance = PathRegistry.get_instance()  # get instance first
    if llm:
        all_tools += agents.load_tools(["llm-math"], llm)
        # all_tools += [PythonREPLTool()]
        all_tools += [
            ModifyBaseSimulationScriptTool(
                path_registry=path_instance, llm=llm, modifysim_no_run=modifysim_no_run
            ),
        ]
        if path_instance.ckpt_papers:
            all_tools += [Scholar2ResultLLM(llm=llm, path_registry=path_instance)]
        if human:
            all_tools += [agents.load_tools(["human"], llm)[0]]

    # add base tools
    base_tools = [
        SummarizeProteinStructure(path_registry=path_instance),
        ComputeAcylindricity(path_registry=path_instance),
        ComputeAsphericity(path_registry=path_instance),
        ComputeDSSP(path_registry=path_instance),
        ComputeGyrationTensor(path_registry=path_instance),
        ComputeRelativeShapeAntisotropy(path_registry=path_instance),
        ComputeAngles(path_registry=path_instance),
        CleaningToolFunction(path_registry=path_instance),
        ComputeLPRMSD(path_registry=path_instance),
        ComputeRMSD(path_registry=path_instance),
        ComputeRMSF(path_registry=path_instance),
        ContactsTool(path_registry=path_instance),
        DistanceMatrixTool(path_registry=path_instance),
        HydrogenBondTool(path_registry=path_instance),
        ListRegistryPaths(path_registry=path_instance),
        MomentOfInertia(path_registry=path_instance),
        PackMolTool(path_registry=path_instance),
        PCATool(path_registry=path_instance),
        PPIDistance(path_registry=path_instance),
        ProteinName2PDBTool(path_registry=path_instance),
        RadiusofGyrationTool(path_registry=path_instance),
        RDFTool(path_registry=path_instance),
        SetUpandRunFunction(path_registry=path_instance),
        SimulationOutputFigures(path_registry=path_instance),
        SmallMolPDB(path_registry=path_instance),
        SolventAccessibleSurfaceArea(path_registry=path_instance),
        VisualizeProtein(path_registry=path_instance),
        MapProteinRepresentation(),
        UniprotID2Name(),
        GetBindingSites(),
        GetActiveSites(),
        GetRelevantSites(),
        GetAllKnownSites(),
        GetProteinFunction(),
        GetProteinAssociatedKeywords(),
        GetAllSequences(),
        GetInteractions(),
        GetSubunitStructure(),
        GetSequenceInfo(),
        GetPDBProcessingInfo(),
        GetPDB3DInfo(),
        GetTurnsBetaSheetsHelices(),
        GetUniprotID(),
        GetGeneNames(),
        GetKineticProperties(),
    ]

    all_tools += base_tools
    return all_tools


def get_relevant_tools(query, llm: BaseLanguageModel, top_k_tools=15, human=False):
    """
    Get most relevant tools for the query using vector similarity search.
    Query and tools are vectorized using either OpenAI embeddings or TF-IDF.

    If an OpenAI API key is available, it uses embeddings for a more
    sophisticated search. Otherwise, it falls back to using TF-IDF for
    simpler, term-based matching.

    Returns:
    - A list of the most relevant tools, or None if no tools are found.
    """

    all_tools = make_all_tools(llm, human=human)
    if not all_tools:
        return None

    tool_texts = [f"{tool.name} {tool.description}" for tool in all_tools]

    # convert texts to vectors
    if "OPENAI_API_KEY" in os.environ:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        try:
            tool_vectors = np.array(embeddings.embed_documents(tool_texts))
            query_vector = np.array(embeddings.embed_query(query)).reshape(1, -1)
        except Exception as e:
            print(f"Error generating embeddings for tool retrieval: {e}")
            return None
    else:
        vectorizer = TfidfVectorizer()
        tool_vectors = vectorizer.fit_transform(tool_texts)
        query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, tool_vectors).flatten()
    k = min(max(top_k_tools, 1), len(all_tools))
    if k == 0:
        return None
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    retrieved_tools = [all_tools[i] for i in top_k_indices]

    return retrieved_tools
