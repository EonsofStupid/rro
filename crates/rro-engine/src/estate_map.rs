//! Render an estate as one connectome: memory, nodes, warp points, connectors,
//! tags, shapes, and trends in a single graph.
//!
//! This is the relationship model made visible: the estate at the center; the
//! **nodes** it federates (each with its layer-2 a2a warp points); the
//! **connectors** operators shared and how much each has fed the estate; the
//! **tags**, **shapes**, and **trends** describing what the memory holds and
//! how it is moving.

use connectome::{ConnectomeGraph, EdgeKind, NodeKind};
use connxism::Estate;
use rro_core::Result;

/// Build the estate-wide connectome.
pub fn estate_map(estate: &Estate) -> Result<ConnectomeGraph> {
    let mut g = ConnectomeGraph::new();
    let info = estate.info();

    let estate_id = format!("estate:{}", info.id);
    g.node(&estate_id, NodeKind::Estate, &info.name, None);

    // Nodes and their warp points.
    for node in estate.nodes()? {
        let nid = format!("node:{}", node.id);
        g.node(&nid, NodeKind::NetNode, &node.name, None);
        g.edge(&estate_id, &nid, EdgeKind::Member, 1.0);
        for (i, warp) in node.warp_points.iter().enumerate() {
            let wid = format!("warp:{}:{i}", node.id);
            let label = format!("{:?} {}", warp.transport, warp.address).to_lowercase();
            g.node(&wid, NodeKind::NetNode, label, None);
            g.edge(&nid, &wid, EdgeKind::Warp, 1.0);
        }
    }

    // Connectors, weighted by how much they've fed the estate.
    for conn in estate.connectors()? {
        let cid = format!("conn:{}", conn.id);
        let label = format!("{} ({})", conn.name, conn.provider);
        g.node(&cid, NodeKind::Connector, label, None);
        g.edge(&estate_id, &cid, EdgeKind::Member, 1.0);
        g.edge(
            &cid,
            &estate_id,
            EdgeKind::Fed,
            conn.sync.docs_synced as f32,
        );
    }

    // Tags with their membership counts.
    for tag in estate.tags()? {
        let tid = format!("tag:{tag}");
        let count = estate.tag_count(&tag)? as f32;
        g.node(&tid, NodeKind::Tag, format!("#{tag}"), Some(count));
        g.edge(&estate_id, &tid, EdgeKind::Tagged, count);
    }

    // The shape census.
    for (shape, count) in estate.shapes()? {
        if count == 0 {
            continue;
        }
        let sid = format!("shape:{shape}");
        let label = if shape.is_empty() {
            "(no fields)".to_string()
        } else {
            shape.clone()
        };
        g.node(&sid, NodeKind::Shape, label, Some(count as f32));
        g.edge(&estate_id, &sid, EdgeKind::Member, count as f32);
    }

    // Trends: latest value per metric.
    for metric in estate.trend_metrics()? {
        let series = estate.trend(&metric)?;
        if let Some(last) = series.last() {
            let tid = format!("trend:{metric}");
            let label = format!("{metric} = {:.1}", last.value);
            g.node(&tid, NodeKind::Trend, label, Some(last.value as f32));
            g.edge(&estate_id, &tid, EdgeKind::Member, 1.0);
        }
    }

    Ok(g)
}
