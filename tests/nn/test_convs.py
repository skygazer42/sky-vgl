import pytest
import torch

from vgl import Graph
from vgl.nn.conv.agnn import AGNNConv
from vgl.nn.conv.cheb import ChebConv
from vgl.nn.conv.clustergcn import ClusterGCNConv
from vgl.nn.conv.appnp import APPNPConv
from vgl.nn.conv.arma import ARMAConv
from vgl.nn.conv.antisymmetric import AntiSymmetricConv
from vgl.nn.conv.cg import CGConv
from vgl.nn.conv.dna import DNAConv
from vgl.nn.conv.fagcn import FAGCNConv
from vgl.nn.conv.fa import FAConv
from vgl.nn.conv.edgeconv import EdgeConv
from vgl.nn.conv.film import FiLMConv
from vgl.nn.conv.feast import FeaStConv
from vgl.nn.conv.generalconv import GeneralConv
from vgl.nn.conv.gcn import GCNConv
from vgl.nn.conv.gatv2 import GATv2Conv
from vgl.nn.conv.gatedgcn import GatedGCNConv
from vgl.nn.conv.gatedgraph import GatedGraphConv
from vgl.nn.conv.gen import GENConv
from vgl.nn.conv.gine import GINEConv
from vgl.nn.conv.gin import GINConv
from vgl.nn.conv.gprgnn import GPRGNNConv
from vgl.nn.conv.gcn2 import GCN2Conv
from vgl.nn.conv.gmm import GMMConv
from vgl.nn.conv.han import HANConv
from vgl.nn.conv.lightgcn import LightGCNConv
from vgl.nn.conv.mixhop import MixHopConv
from vgl.nn.conv.mfconv import MFConv
from vgl.nn.conv.bern import BernConv
from vgl.nn.conv.dagnn import DAGNNConv
from vgl.nn.conv.egconv import EGConv
from vgl.nn.conv.graphconv import GraphConv
from vgl.nn.conv.h2gcn import H2GCNConv
from vgl.nn.conv.heat import HEATConv
from vgl.nn.conv.hgt import HGTConv
from vgl.nn.conv.leconv import LEConv
from vgl.nn.conv.lg import LGConv
from vgl.nn.conv.resgated import ResGatedGraphConv
from vgl.nn.conv.pna import PNAConv
from vgl.nn.conv.pdn import PDNConv
from vgl.nn.conv.pointnet import PointNetConv
from vgl.nn.conv.pointtransformer import PointTransformerConv
from vgl.nn.conv.sg import SGConv
from vgl.nn.conv.sage import SAGEConv
from vgl.nn.conv.simple import SimpleConv
from vgl.nn.conv.spline import SplineConv
from vgl.nn.conv.ssg import SSGConv
from vgl.nn.conv.supergat import SuperGATConv
from vgl.nn.conv.tag import TAGConv
from vgl.nn.conv.transformer import TransformerConv
from vgl.nn.conv.twirls import TWIRLSConv
from vgl.nn.conv.wlconv import WLConvContinuous
from vgl.nn.conv.dirgnn import DirGNNConv
from vgl.nn.conv.nnconv import ECConv, NNConv
from vgl.nn.conv.rgcn import RGCNConv
from vgl.nn.conv.rgat import RGATConv


def _homo_graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        x=torch.randn(3, 4),
        pos=torch.randn(3, 2),
        edge_data={
            "edge_attr": torch.randn(3, 2),
            "pseudo": torch.rand(3, 2),
        },
    )


def _hetero_graph():
    return Graph.hetero(
        nodes={"paper": {"x": torch.randn(2, 4)}, "author": {"x": torch.randn(2, 4)}},
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1], [1, 0]]),
                "edge_attr": torch.randn(2, 2),
            },
            ("paper", "written_by", "author"): {
                "edge_index": torch.tensor([[1, 0], [0, 1]]),
                "edge_attr": torch.randn(2, 2),
            },
        },
    )


def test_gcn_conv_accepts_graph_input():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
    )
    conv = GCNConv(in_channels=4, out_channels=3)

    out = conv(graph)

    assert out.shape == (2, 3)


def test_rgcn_conv_accepts_hetero_graph_input():
    graph = _hetero_graph()
    conv = RGCNConv(
        in_channels=4,
        out_channels=3,
        node_types=graph.schema.node_types,
        relation_types=graph.schema.edge_types,
    )

    out = conv(graph)

    assert set(out) == {"author", "paper"}
    assert out["author"].shape == (2, 3)
    assert out["paper"].shape == (2, 3)


def test_hgt_conv_accepts_hetero_graph_input():
    graph = _hetero_graph()
    conv = HGTConv(
        in_channels=4,
        out_channels=3,
        node_types=graph.schema.node_types,
        relation_types=graph.schema.edge_types,
        heads=2,
    )

    out = conv(graph)

    assert set(out) == {"author", "paper"}
    assert out["author"].shape == (2, 3)
    assert out["paper"].shape == (2, 3)


def test_heat_conv_accepts_hetero_graph_input():
    graph = _hetero_graph()
    conv = HEATConv(
        in_channels=4,
        out_channels=3,
        node_types=graph.schema.node_types,
        relation_types=graph.schema.edge_types,
        edge_channels=2,
        heads=2,
    )

    out = conv(graph)

    assert set(out) == {"author", "paper"}
    assert out["author"].shape == (2, 3)
    assert out["paper"].shape == (2, 3)


def test_rgat_conv_accepts_hetero_graph_input():
    graph = _hetero_graph()
    conv = RGATConv(
        in_channels=4,
        out_channels=3,
        node_types=graph.schema.node_types,
        relation_types=graph.schema.edge_types,
        edge_channels=2,
        heads=2,
    )

    out = conv(graph)

    assert set(out) == {"author", "paper"}
    assert out["author"].shape == (2, 3)
    assert out["paper"].shape == (2, 3)


def test_han_conv_accepts_hetero_graph_input():
    graph = _hetero_graph()
    conv = HANConv(
        in_channels=4,
        out_channels=3,
        node_types=graph.schema.node_types,
        relation_types=graph.schema.edge_types,
    )

    out = conv(graph)

    assert set(out) == {"author", "paper"}
    assert out["author"].shape == (2, 3)
    assert out["paper"].shape == (2, 3)


def test_sage_conv_accepts_x_and_edge_index():
    x = torch.randn(2, 4)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    conv = SAGEConv(in_channels=4, out_channels=3)

    out = conv(x, edge_index)

    assert out.shape == (2, 3)


def test_gin_conv_accepts_graph_input():
    conv = GINConv(in_channels=4, out_channels=3)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_gin_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = GINConv(in_channels=4, out_channels=3)

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_cgconv_accepts_graph_input_with_edge_features():
    conv = CGConv(channels=4, edge_channels=2, aggr="mean")

    out = conv(_homo_graph())

    assert out.shape == (3, 4)


def test_dnaconv_accepts_graph_input_with_history_tensor():
    graph = _homo_graph()
    history = torch.stack([graph.x, graph.x + 1.0, graph.x - 1.0], dim=1)
    conv = DNAConv(channels=4, heads=2, dropout=0.0)

    out = conv(graph, history=history)

    assert out.shape == (3, 4)


def test_dnaconv_accepts_history_tensor_and_edge_index():
    graph = _homo_graph()
    history = torch.stack([graph.x, graph.x * 0.5], dim=1)
    conv = DNAConv(channels=4, heads=2, dropout=0.0)

    out = conv(history, graph.edge_index)

    assert out.shape == (3, 4)


def test_nnconv_accepts_graph_input_with_edge_features():
    conv = NNConv(in_channels=4, out_channels=3, edge_channels=2)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_ecconv_is_nnconv_compatible():
    conv = ECConv(in_channels=4, out_channels=3, edge_channels=2)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_gine_conv_accepts_graph_input_with_edge_features():
    conv = GINEConv(in_channels=4, out_channels=3, edge_channels=2)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_gmm_conv_accepts_graph_input_with_pseudo_coordinates():
    conv = GMMConv(in_channels=4, out_channels=3, dim=2, kernel_size=3)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_spline_conv_accepts_graph_input_with_pseudo_coordinates():
    conv = SplineConv(in_channels=4, out_channels=3, dim=2, kernel_size=3)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_faconv_accepts_graph_input_with_reference_features():
    graph = _homo_graph()
    conv = FAConv(channels=4, eps=0.1)

    out = conv(graph, x0=graph.x)

    assert out.shape == (3, 4)


def test_gatedgcn_conv_accepts_graph_input_with_edge_features():
    conv = GatedGCNConv(in_channels=4, out_channels=3, edge_channels=2)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_pdn_conv_accepts_x_edge_index_and_edge_features():
    graph = _homo_graph()
    conv = PDNConv(in_channels=4, out_channels=3, edge_channels=2, add_self_loops=False)

    out = conv(graph.x, graph.edge_index, edge_attr=graph.edata["edge_attr"])

    assert out.shape == (3, 3)


def test_pointnet_conv_accepts_graph_input_with_pos():
    conv = PointNetConv(in_channels=4, out_channels=3, pos_channels=2)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_point_transformer_conv_accepts_x_edge_index_and_pos():
    graph = _homo_graph()
    conv = PointTransformerConv(in_channels=4, out_channels=3, pos_channels=2)

    out = conv(graph.x, graph.edge_index, pos=graph.pos)

    assert out.shape == (3, 3)


def test_twirls_conv_accepts_graph_input():
    conv = TWIRLSConv(in_channels=4, out_channels=3, steps=3, alpha=0.2)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_twirls_conv_accepts_x_and_edge_index():
    graph = _homo_graph()
    conv = TWIRLSConv(in_channels=4, out_channels=3, steps=2, alpha=0.2)

    out = conv(graph.x, graph.edge_index)

    assert out.shape == (3, 3)


def test_gatv2_conv_respects_head_output_shapes():
    graph = _homo_graph()

    concat_conv = GATv2Conv(in_channels=4, out_channels=3, heads=2, concat=True)
    mean_conv = GATv2Conv(in_channels=4, out_channels=3, heads=2, concat=False)

    concat_out = concat_conv(graph)
    mean_out = mean_conv(graph)

    assert concat_out.shape == (3, 6)
    assert mean_out.shape == (3, 3)


def test_appnp_conv_accepts_graph_input():
    conv = APPNPConv(in_channels=4, out_channels=3, steps=2, alpha=0.1)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_tag_conv_accepts_graph_input():
    conv = TAGConv(in_channels=4, out_channels=3, k=2)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_tag_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = TAGConv(in_channels=4, out_channels=3, k=2)

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_sg_conv_accepts_graph_input():
    conv = SGConv(in_channels=4, out_channels=3, k=2)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_cheb_conv_accepts_graph_input():
    conv = ChebConv(in_channels=4, out_channels=3, k=3)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_agnn_conv_accepts_graph_input():
    conv = AGNNConv(channels=4)

    out = conv(_homo_graph())

    assert out.shape == (3, 4)


def test_agnn_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = AGNNConv(channels=4)

    out = conv(x, edge_index)

    assert out.shape == (3, 4)


def test_lightgcn_conv_accepts_graph_input():
    conv = LightGCNConv()

    out = conv(_homo_graph())

    assert out.shape == (3, 4)


def test_lightgcn_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = LightGCNConv()

    out = conv(x, edge_index)

    assert out.shape == (3, 4)


def test_lgconv_accepts_graph_input():
    conv = LGConv()

    out = conv(_homo_graph())

    assert out.shape == (3, 4)


def test_lgconv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = LGConv()

    out = conv(x, edge_index)

    assert out.shape == (3, 4)


def test_lgconv_without_normalization_matches_additive_propagation():
    x = torch.tensor([[1.0], [2.0], [4.0]])
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]])
    conv = LGConv(normalize=False)

    out = conv(x, edge_index)

    expected = torch.tensor([[4.0], [1.0], [3.0]])
    assert torch.allclose(out, expected)


def test_fagcn_conv_accepts_graph_input():
    conv = FAGCNConv(channels=4, eps=0.1)

    out = conv(_homo_graph())

    assert out.shape == (3, 4)


def test_arma_conv_accepts_graph_input():
    conv = ARMAConv(channels=4, stacks=2, layers=2, alpha=0.1)

    out = conv(_homo_graph())

    assert out.shape == (3, 4)


def test_arma_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = ARMAConv(channels=4, stacks=2, layers=2, alpha=0.1)

    out = conv(x, edge_index)

    assert out.shape == (3, 4)


def test_gprgnn_conv_accepts_graph_input():
    conv = GPRGNNConv(channels=4, steps=3, alpha=0.1)

    out = conv(_homo_graph())

    assert out.shape == (3, 4)


def test_gprgnn_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = GPRGNNConv(channels=4, steps=3, alpha=0.1)

    out = conv(x, edge_index)

    assert out.shape == (3, 4)


def test_mixhop_conv_accepts_graph_input():
    conv = MixHopConv(in_channels=4, out_channels=3, powers=(0, 1, 2))

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_mixhop_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = MixHopConv(in_channels=4, out_channels=3, powers=(0, 1, 2))

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_bern_conv_accepts_graph_input():
    conv = BernConv(channels=4, steps=3)

    out = conv(_homo_graph())

    assert out.shape == (3, 4)


def test_bern_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = BernConv(channels=4, steps=3)

    out = conv(x, edge_index)

    assert out.shape == (3, 4)


def test_ssg_conv_accepts_graph_input():
    conv = SSGConv(channels=4, steps=3, alpha=0.1)

    out = conv(_homo_graph())

    assert out.shape == (3, 4)


def test_ssg_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = SSGConv(channels=4, steps=3, alpha=0.1)

    out = conv(x, edge_index)

    assert out.shape == (3, 4)


def test_dagnn_conv_accepts_graph_input():
    conv = DAGNNConv(channels=4, steps=3)

    out = conv(_homo_graph())

    assert out.shape == (3, 4)


def test_dagnn_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = DAGNNConv(channels=4, steps=3)

    out = conv(x, edge_index)

    assert out.shape == (3, 4)


def test_h2gcn_conv_accepts_graph_input():
    conv = H2GCNConv(in_channels=4, out_channels=3)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_h2gcn_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = H2GCNConv(in_channels=4, out_channels=3)

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_egconv_accepts_graph_input():
    conv = EGConv(in_channels=4, out_channels=3, aggregators=("sum", "mean", "max"))

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_egconv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = EGConv(in_channels=4, out_channels=3, aggregators=("sum", "mean", "max"))

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_gcn2_conv_accepts_graph_input_with_x0():
    graph = _homo_graph()
    conv = GCN2Conv(channels=4, alpha=0.1, theta=1.0, layer=1)

    out = conv(graph, x0=graph.x)

    assert out.shape == (3, 4)


def test_gcn2_conv_accepts_x_edge_index_and_x0():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = GCN2Conv(channels=4, alpha=0.1, theta=1.0, layer=1)

    out = conv(x, edge_index, x0=x)

    assert out.shape == (3, 4)


def test_gcn2_conv_requires_explicit_x0():
    conv = GCN2Conv(channels=4, alpha=0.1, theta=1.0, layer=1)

    with pytest.raises(ValueError, match="x0"):
        conv(_homo_graph())


def test_graph_conv_accepts_graph_input():
    conv = GraphConv(in_channels=4, out_channels=3)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_graph_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = GraphConv(in_channels=4, out_channels=3)

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_leconv_accepts_graph_input():
    conv = LEConv(in_channels=4, out_channels=3)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_leconv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = LEConv(in_channels=4, out_channels=3)

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_resgated_graph_conv_accepts_graph_input():
    conv = ResGatedGraphConv(in_channels=4, out_channels=3)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_resgated_graph_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = ResGatedGraphConv(in_channels=4, out_channels=3)

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_gated_graph_conv_accepts_graph_input():
    conv = GatedGraphConv(channels=4, steps=2)

    out = conv(_homo_graph())

    assert out.shape == (3, 4)


def test_gated_graph_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = GatedGraphConv(channels=4, steps=2)

    out = conv(x, edge_index)

    assert out.shape == (3, 4)


def test_clustergcn_conv_accepts_graph_input():
    conv = ClusterGCNConv(in_channels=4, out_channels=3, diag_lambda=0.0)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_clustergcn_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = ClusterGCNConv(in_channels=4, out_channels=3, diag_lambda=0.0)

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_gen_conv_accepts_graph_input():
    conv = GENConv(in_channels=4, out_channels=3, aggr="softmax", beta=1.0)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_gen_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = GENConv(in_channels=4, out_channels=3, aggr="softmax", beta=1.0)

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_film_conv_accepts_graph_input():
    conv = FiLMConv(in_channels=4, out_channels=3)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_film_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = FiLMConv(in_channels=4, out_channels=3)

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_simple_conv_accepts_graph_input():
    conv = SimpleConv(aggr="mean")

    out = conv(_homo_graph())

    assert out.shape == (3, 4)


def test_simple_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = SimpleConv(aggr="mean")

    out = conv(x, edge_index)

    assert out.shape == (3, 4)


def test_edge_conv_accepts_graph_input():
    conv = EdgeConv(in_channels=4, out_channels=3, aggr="max")

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_edge_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = EdgeConv(in_channels=4, out_channels=3, aggr="max")

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_feast_conv_accepts_graph_input():
    conv = FeaStConv(in_channels=4, out_channels=3, heads=2)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_feast_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = FeaStConv(in_channels=4, out_channels=3, heads=2)

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_mfconv_accepts_graph_input():
    conv = MFConv(in_channels=4, out_channels=3, max_degree=4)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_mfconv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = MFConv(in_channels=4, out_channels=3, max_degree=4)

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_pna_conv_accepts_graph_input():
    conv = PNAConv(
        in_channels=4,
        out_channels=3,
        aggregators=("sum", "mean", "max"),
        scalers=("identity", "amplification", "attenuation"),
    )

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_pna_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = PNAConv(
        in_channels=4,
        out_channels=3,
        aggregators=("sum", "mean", "max"),
        scalers=("identity", "amplification", "attenuation"),
    )

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_general_conv_accepts_graph_input():
    conv = GeneralConv(
        in_channels=4,
        out_channels=4,
        aggr="add",
        heads=2,
        attention=True,
    )

    out = conv(_homo_graph())

    assert out.shape == (3, 4)


def test_general_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = GeneralConv(
        in_channels=4,
        out_channels=4,
        aggr="add",
        heads=2,
        attention=True,
    )

    out = conv(x, edge_index)

    assert out.shape == (3, 4)


def test_antisymmetric_conv_accepts_graph_input():
    conv = AntiSymmetricConv(channels=4, num_iters=2, epsilon=0.1, gamma=0.1)

    out = conv(_homo_graph())

    assert out.shape == (3, 4)


def test_antisymmetric_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = AntiSymmetricConv(channels=4, num_iters=2, epsilon=0.1, gamma=0.1)

    out = conv(x, edge_index)

    assert out.shape == (3, 4)


def test_pna_conv_rejects_unsupported_aggregator():
    with pytest.raises(ValueError, match="aggregator"):
        PNAConv(in_channels=4, out_channels=3, aggregators=("sum", "median"))


def test_pna_conv_rejects_unsupported_scaler():
    with pytest.raises(ValueError, match="scaler"):
        PNAConv(in_channels=4, out_channels=3, scalers=("identity", "inverse"))


def test_general_conv_rejects_unsupported_aggr():
    with pytest.raises(ValueError, match="aggr"):
        GeneralConv(in_channels=4, out_channels=4, aggr="softmax")


def test_general_conv_rejects_invalid_heads():
    with pytest.raises(ValueError, match="heads|out_channels"):
        GeneralConv(in_channels=4, out_channels=3, heads=2)


def test_antisymmetric_conv_rejects_unsupported_activation():
    with pytest.raises(ValueError, match="act|activation"):
        AntiSymmetricConv(channels=4, act="gelu")


def test_transformer_conv_accepts_graph_input():
    conv = TransformerConv(in_channels=4, out_channels=3, heads=2, concat=False)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_transformer_conv_avoids_tensor_item_in_edge_softmax(monkeypatch):
    conv = TransformerConv(in_channels=4, out_channels=3, heads=2, concat=False)

    def fail_item(self):
        raise AssertionError("homo edge_softmax should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_transformer_conv_avoids_tensor_int_in_edge_softmax(monkeypatch):
    conv = TransformerConv(in_channels=4, out_channels=3, heads=2, concat=False)

    def fail_int(self):
        raise AssertionError("homo edge_softmax should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_transformer_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = TransformerConv(in_channels=4, out_channels=3, heads=2, concat=False)

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_transformer_conv_concat_shape():
    conv = TransformerConv(in_channels=4, out_channels=3, heads=2, concat=True)

    out = conv(_homo_graph())

    assert out.shape == (3, 6)


def test_transformer_conv_beta_path():
    conv = TransformerConv(
        in_channels=4,
        out_channels=3,
        heads=2,
        concat=False,
        beta=True,
        root_weight=True,
    )

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_transformer_conv_rejects_invalid_heads():
    with pytest.raises(ValueError, match="heads"):
        TransformerConv(in_channels=4, out_channels=3, heads=0)


def test_transformer_conv_rejects_invalid_dropout():
    with pytest.raises(ValueError, match="dropout"):
        TransformerConv(in_channels=4, out_channels=3, dropout=1.5)


def test_wlconv_continuous_accepts_graph_input():
    conv = WLConvContinuous()

    out = conv(_homo_graph())

    assert out.shape == (3, 4)


def test_wlconv_continuous_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = WLConvContinuous()

    out = conv(x, edge_index)

    assert out.shape == (3, 4)


def test_supergat_conv_accepts_graph_input():
    conv = SuperGATConv(in_channels=4, out_channels=3, heads=2, concat=False)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_supergat_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = SuperGATConv(in_channels=4, out_channels=3, heads=2, concat=False)

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_supergat_conv_concat_shape():
    conv = SuperGATConv(in_channels=4, out_channels=3, heads=2, concat=True)

    out = conv(_homo_graph())

    assert out.shape == (3, 6)


def test_supergat_conv_exposes_attention_loss():
    conv = SuperGATConv(in_channels=4, out_channels=3, heads=2, concat=False)

    conv(_homo_graph())
    loss = conv.get_attention_loss()

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_supergat_conv_rejects_invalid_heads():
    with pytest.raises(ValueError, match="heads"):
        SuperGATConv(in_channels=4, out_channels=3, heads=0)


def test_supergat_conv_rejects_invalid_dropout():
    with pytest.raises(ValueError, match="dropout"):
        SuperGATConv(in_channels=4, out_channels=3, dropout=1.5)


def test_supergat_conv_rejects_invalid_attention_type():
    with pytest.raises(ValueError, match="attention_type"):
        SuperGATConv(in_channels=4, out_channels=3, attention_type="BAD")


def test_dirgnn_conv_accepts_graph_input():
    conv = DirGNNConv(GraphConv(in_channels=4, out_channels=3), alpha=0.5, root_weight=True)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_dirgnn_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = DirGNNConv(GraphConv(in_channels=4, out_channels=3), alpha=0.5, root_weight=True)

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_dirgnn_conv_alpha_one_matches_forward_branch():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    base = GraphConv(in_channels=4, out_channels=3)
    conv = DirGNNConv(base, alpha=1.0, root_weight=False)

    expected = base(x, edge_index)
    out = conv(x, edge_index)

    assert torch.allclose(out, expected)


def test_dirgnn_conv_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        DirGNNConv(GraphConv(in_channels=4, out_channels=3), alpha=1.5)


def test_dirgnn_conv_rejects_unsupported_base_contract():
    class NeedsExtraArg(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.out_channels = 4

        def forward(self, x, edge_index, extra):
            return x

    with pytest.raises(ValueError, match="forward|extra|runtime"):
        DirGNNConv(NeedsExtraArg())


@pytest.mark.parametrize(
    ("conv_cls", "kwargs"),
    [
        (GINConv, {"in_channels": 4, "out_channels": 3}),
        (GATv2Conv, {"in_channels": 4, "out_channels": 3}),
        (APPNPConv, {"in_channels": 4, "out_channels": 3}),
        (TAGConv, {"in_channels": 4, "out_channels": 3}),
        (SGConv, {"in_channels": 4, "out_channels": 3}),
        (ChebConv, {"in_channels": 4, "out_channels": 3}),
        (AGNNConv, {"channels": 4}),
        (LightGCNConv, {}),
        (LGConv, {}),
        (FAGCNConv, {"channels": 4}),
        (ARMAConv, {"channels": 4}),
        (GPRGNNConv, {"channels": 4}),
        (MixHopConv, {"in_channels": 4, "out_channels": 3}),
        (BernConv, {"channels": 4}),
        (SSGConv, {"channels": 4}),
        (DAGNNConv, {"channels": 4}),
        (GCN2Conv, {"channels": 4}),
        (GraphConv, {"in_channels": 4, "out_channels": 3}),
        (H2GCNConv, {"in_channels": 4, "out_channels": 3}),
        (EGConv, {"in_channels": 4, "out_channels": 3}),
        (LEConv, {"in_channels": 4, "out_channels": 3}),
        (ResGatedGraphConv, {"in_channels": 4, "out_channels": 3}),
        (GatedGraphConv, {"channels": 4}),
        (ClusterGCNConv, {"in_channels": 4, "out_channels": 3}),
        (GENConv, {"in_channels": 4, "out_channels": 3}),
        (FiLMConv, {"in_channels": 4, "out_channels": 3}),
        (SimpleConv, {}),
        (EdgeConv, {"in_channels": 4, "out_channels": 3}),
        (FeaStConv, {"in_channels": 4, "out_channels": 3}),
        (MFConv, {"in_channels": 4, "out_channels": 3}),
        (PNAConv, {"in_channels": 4, "out_channels": 3}),
        (GeneralConv, {"in_channels": 4, "out_channels": 4}),
        (AntiSymmetricConv, {"channels": 4}),
        (TransformerConv, {"in_channels": 4, "out_channels": 3}),
        (WLConvContinuous, {}),
        (SuperGATConv, {"in_channels": 4, "out_channels": 3}),
        (DirGNNConv, {"conv": GraphConv(in_channels=4, out_channels=3)}),
    ],
)
def test_new_homo_convs_reject_hetero_graph_input(conv_cls, kwargs):
    conv = conv_cls(**kwargs)

    with pytest.raises(ValueError, match="homogeneous"):
        conv(_hetero_graph())
