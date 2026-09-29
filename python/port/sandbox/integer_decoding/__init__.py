"""Integer-copy decoding set aside with its numbers (#362).

What the sim audit compared the pipeline's decode against and did not
adopt: CalicoST's own decoders, called as CalicoST calls them
(`calicost_decoders`), and two per-bin read-depth summaries that cannot filter
out a CNA (`rdr_summary`). Installed by nothing.
"""
