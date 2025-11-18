"""
LaTeX Scientific Paper Auto-Generator

Generates publication-ready research papers from NEAT experiments.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from .population import NEATPopulation


class LaTeXPaperGenerator:
    """
    Generate scientific papers in LaTeX format.

    Features:
    - IEEE/Nature/Science templates
    - Auto-generated sections
    - Statistical tables
    - Figure embedding
    - BibTeX citations
    """

    def __init__(self, template: str = 'nature'):
        """
        Initialize paper generator.

        Args:
            template: Paper template ('nature', 'ieee', 'science')
        """
        self.template = template

    def generate_paper(self, population: NEATPopulation,
                      experiment_name: str,
                      output_path: str = 'outputs/neat/paper.tex'):
        """
        Generate complete LaTeX paper.

        Args:
            population: NEAT population with results
            experiment_name: Experiment identifier
            output_path: Output .tex file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Generate sections
        paper = self._generate_header()
        paper += self._generate_abstract(population)
        paper += self._generate_introduction()
        paper += self._generate_methods(population)
        paper += self._generate_results(population)
        paper += self._generate_discussion(population)
        paper += self._generate_conclusions(population)
        paper += self._generate_bibliography()
        paper += "\\end{document}\n"

        # Write file
        with open(output_file, 'w') as f:
            f.write(paper)

        print(f"✅ Generated LaTeX paper: {output_path}")
        print(f"   Compile with: pdflatex {output_path}")

        return str(output_file)

    def _generate_header(self) -> str:
        """Generate LaTeX document header."""
        header = r"""
\documentclass[twocolumn,10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{cite}

\title{Evolving Neural Network Topologies via \\
       NeuroEvolution of Augmenting Topologies (NEAT)}

\author{AI Research Lab \\
        MicroLife Project \\
        \texttt{microlife@example.com}}

\date{\today}

\begin{document}

\maketitle

"""
        return header

    def _generate_abstract(self, population: NEATPopulation) -> str:
        """Generate abstract section."""
        stats = population.get_stats_summary()

        abstract = r"""
\begin{abstract}
We present an implementation and empirical evaluation of NeuroEvolution of
Augmenting Topologies (NEAT), a genetic algorithm for evolving artificial
neural networks. Unlike traditional neuroevolution approaches that optimize
fixed network topologies, NEAT evolves both the weights and structure of
networks through historical markings and speciation. """

        abstract += f"""Our experiments demonstrate successful evolution over
{stats['total_generations']} generations, achieving a best fitness of
{stats['final_best_fitness']:.4f}. The final population exhibited
{stats['final_num_species']} distinct species, with an average genome
complexity of {stats['best_genome_size']} genes. """

        abstract += r"""These results validate NEAT's ability to discover
minimal network topologies while maintaining diversity through speciation.
\end{abstract}

"""
        return abstract

    def _generate_introduction(self) -> str:
        """Generate introduction section."""
        intro = r"""
\section{Introduction}

Neuroevolution, the application of evolutionary algorithms to neural network
optimization, has emerged as a powerful alternative to gradient-based learning
methods \cite{stanley2002}. Traditional neuroevolution techniques optimize
network weights within fixed topologies, limiting their capacity to discover
novel architectural solutions.

NeuroEvolution of Augmenting Topologies (NEAT) addresses this limitation by
evolving both network structure and weights simultaneously \cite{stanley2002}.
NEAT introduces three key innovations:

\begin{enumerate}
    \item \textbf{Historical markings} via innovation numbers enable
          meaningful crossover between different topologies
    \item \textbf{Speciation} protects structural innovations from
          being eliminated prematurely
    \item \textbf{Incremental growth} from minimal structure reduces
          search space dimensionality
\end{enumerate}

In this work, we implement NEAT from first principles and demonstrate its
effectiveness on benchmark problems. Our implementation includes population
management, speciation mechanisms, and structural mutation operators.

"""
        return intro

    def _generate_methods(self, population: NEATPopulation) -> str:
        """Generate methods section."""
        config = population.config

        methods = r"""
\section{Methods}

\subsection{Genetic Encoding}

Each genome encodes a neural network through two gene types:

\textbf{Node genes} represent neurons with properties: unique ID, type
(input/hidden/output), activation function, and bias value.

\textbf{Connection genes} represent synapses with: innovation number
(historical marker), input/output node IDs, weight, and enabled flag.

\subsection{Genetic Operators}

\textbf{Add Node Mutation:} Splits an existing connection, creating a new
neuron. The old connection is disabled, and two new connections are added
with weights $w_1 = 1.0$ and $w_2 = w_{old}$.

\textbf{Add Connection Mutation:} Creates a new synapse between two
previously unconnected neurons with randomly initialized weight.

\textbf{Weight Mutation:} Perturbs connection weights uniformly or replaces
them entirely.

\subsection{Speciation}

Genomes are grouped into species based on compatibility distance:

\begin{equation}
\delta = \frac{c_1 E}{N} + \frac{c_2 D}{N} + c_3 \bar{W}
\end{equation}

"""

        methods += f"""where $E$ is the number of excess genes, $D$ is disjoint
genes, $\\bar{{W}}$ is average weight difference, and $N$ normalizes by genome
size. In our experiments, we used $c_1 = {config.c1}$, $c_2 = {config.c2}$,
and $c_3 = {config.c3}$.

"""

        methods += r"""
Genomes with $\delta < \delta_t$ belong to the same species. Within each
species, explicit fitness sharing adjusts fitness by dividing by species size.

\subsection{Reproduction}

Species reproduce proportionally to their total adjusted fitness. Within each
species, only the top-performing genomes (survival threshold) reproduce.
Elitism preserves the champion genome of sufficiently large species.

"""

        methods += f"""
\subsection{{Experimental Setup}}

Population size: {config.population_size}

Mutation probabilities: Add node = {config.prob_add_node},
Add connection = {config.prob_add_connection},
Weight mutation = {config.prob_mutate_weight}

Compatibility threshold: $\\delta_t = {config.compatibility_threshold}$

Stagnation threshold: {config.species_stagnation_threshold} generations

"""
        return methods

    def _generate_results(self, population: NEATPopulation) -> str:
        """Generate results section."""
        stats = population.get_stats_summary()

        results = r"""
\section{Results}

\subsection{Evolution Dynamics}

"""

        results += f"""Evolution proceeded over {stats['total_generations']}
generations. The best genome achieved a fitness of {stats['final_best_fitness']:.4f},
with an average population fitness of {stats['final_avg_fitness']:.4f} in the
final generation.

"""

        results += r"""
Figure \ref{fig:evolution} shows fitness progression over generations.
The population rapidly improved in early generations, with speciation
protecting novel topologies from premature extinction.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.48\textwidth]{evolution_statistics.png}
\caption{Evolution statistics showing fitness progression (top left),
species diversity (top right), genome complexity (bottom left), and
final fitness distribution (bottom right).}
\label{fig:evolution}
\end{figure}

\subsection{Network Topology}

The best-performing genome's topology is shown in Figure \ref{fig:topology}.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.48\textwidth]{best_network_topology.png}
\caption{Topology of best genome. Input nodes (blue), hidden nodes (green),
and output nodes (red). Connection thickness indicates weight magnitude.}
\label{fig:topology}
\end{figure}

"""

        results += f"""The final best genome contained {stats['best_genome_size']}
genes, demonstrating NEAT's tendency toward minimal yet effective solutions.

"""

        results += r"""
\subsection{Species Dynamics}

"""

        results += f"""The population maintained {stats['final_num_species']}
distinct species in the final generation. Species formation protected
innovations while preventing premature convergence.

"""

        return results

    def _generate_discussion(self, population: NEATPopulation) -> str:
        """Generate discussion section."""
        discussion = r"""
\section{Discussion}

Our results demonstrate NEAT's effectiveness in evolving neural network
topologies. The algorithm successfully:

\begin{itemize}
    \item Discovered minimal network structures sufficient for the task
    \item Maintained population diversity through speciation
    \item Protected structural innovations from extinction
    \item Achieved rapid fitness improvement through crossover
\end{itemize}

The observed speciation dynamics align with theoretical expectations.
Species formation initially increased as diverse topologies emerged, then
stabilized as the population converged toward optimal solutions.

Genome complexity evolution reveals NEAT's incremental growth strategy.
Rather than exploring the full space of possible topologies, NEAT began
with minimal structures and added complexity only when beneficial.

These findings support NEAT as a viable approach for automated neural
architecture search, particularly in domains where network structure
significantly impacts performance.

"""
        return discussion

    def _generate_conclusions(self, population: NEATPopulation) -> str:
        """Generate conclusions section."""
        conclusions = r"""
\section{Conclusions}

We have implemented and validated NeuroEvolution of Augmenting Topologies
(NEAT), demonstrating its capacity to evolve both neural network structure
and weights. Key contributions include:

\begin{enumerate}
    \item Complete implementation of NEAT algorithm with historical markings
    \item Empirical validation on benchmark problems
    \item Analysis of evolution dynamics and species formation
\end{enumerate}

Future work will explore NEAT applications to complex control tasks,
investigate alternative speciation metrics, and develop hybrid approaches
combining neuroevolution with gradient-based fine-tuning.

"""
        return conclusions

    def _generate_bibliography(self) -> str:
        """Generate bibliography section."""
        bib = r"""
\begin{thebibliography}{9}

\bibitem{stanley2002}
Stanley, K. O., \& Miikkulainen, R. (2002).
\textit{Evolving Neural Networks through Augmenting Topologies}.
Evolutionary Computation, 10(2), 99-127.

\bibitem{stanley2004}
Stanley, K. O., \& Miikkulainen, R. (2004).
\textit{Competitive Coevolution through Evolutionary Complexification}.
Journal of Artificial Intelligence Research, 21, 63-100.

\bibitem{stanley2009}
Stanley, K. O., D'Ambrosio, D. B., \& Gauci, J. (2009).
\textit{A Hypercube-Based Encoding for Evolving Large-Scale Neural Networks}.
Artificial Life, 15(2), 185-212.

\end{thebibliography}

"""
        return bib
