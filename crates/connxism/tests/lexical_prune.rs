//! Sprint 23 gates: the max-score pruned lexical scorer is EXACT — top-k
//! ids and scores equal an in-test brute-force BM25 on selective,
//! common-term, and mixed workloads over a randomized corpus — and the
//! blind df counters track upsert/overwrite/remove precisely.

use connxism::Estate;
use rro_core::text::Analyzer;
use rro_core::{Embedding, Recall, VectorRecord};
use std::collections::HashMap;

fn lcg(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// Randomized corpus: 5 common words (every doc), mid + rare vocabulary,
/// varied tf and lengths.
fn corpus(n: usize, seed: u64) -> Vec<(String, String)> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            let mut words: Vec<String> = vec![
                "alpha".into(),
                "beta".into(),
                "gamma".into(),
                "delta".into(),
                "omega".into(),
            ];
            // Mid-frequency words.
            for _ in 0..(1 + lcg(&mut s) % 4) {
                words.push(format!("mid{}", lcg(&mut s) % 50));
            }
            // Rare words.
            if lcg(&mut s).is_multiple_of(3) {
                words.push(format!("rare{}", lcg(&mut s) % 300));
            }
            // Random repeats (tf variety) + filler (length variety).
            for _ in 0..(lcg(&mut s) % 6) {
                let w = words[(lcg(&mut s) % words.len() as u64) as usize].clone();
                words.push(w);
            }
            for f in 0..(lcg(&mut s) % 10) {
                words.push(format!("filler{i}x{f}"));
            }
            (format!("doc{i:03}"), words.join(" "))
        })
        .collect()
}

/// In-test brute-force BM25 (k1=1.2, b=0.75, +0.5-smoothed clamped idf) —
/// the truth the pruned scorer must reproduce.
fn brute_topk(docs: &[(String, String)], query: &str, k: usize) -> Vec<(String, f32)> {
    let an = Analyzer::default();
    let toks: Vec<Vec<String>> = docs.iter().map(|(_, t)| an.analyze(t)).collect();
    let n = docs.len() as f32;
    let avgdl = (toks.iter().map(Vec::len).sum::<usize>() as f32 / n).max(1.0);
    let mut scores: HashMap<&str, f32> = HashMap::new();
    let mut qseen = std::collections::HashSet::new();
    for qt in an.analyze(query) {
        if !qseen.insert(qt.clone()) {
            continue;
        }
        let df = toks.iter().filter(|d| d.contains(&qt)).count() as f32;
        if df == 0.0 {
            continue;
        }
        let idf = (((n - df + 0.5) / (df + 0.5)) + 1.0).ln().max(0.0);
        for (i, d) in toks.iter().enumerate() {
            let f = d.iter().filter(|w| **w == qt).count() as f32;
            if f == 0.0 {
                continue;
            }
            let dl = d.len() as f32;
            let sc = idf * (f * 2.2) / (f + 1.2 * (1.0 - 0.75 + 0.75 * dl / avgdl));
            *scores.entry(docs[i].0.as_str()).or_insert(0.0) += sc;
        }
    }
    let mut ranked: Vec<(String, f32)> = scores
        .into_iter()
        .map(|(id, sc)| (id.to_string(), sc))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(k);
    ranked
}

#[tokio::test(flavor = "multi_thread")]
async fn pruned_topk_is_exact_on_all_workloads() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "lp").unwrap();
    let recall = estate.recall();
    let docs = corpus(400, 42);
    recall
        .upsert(
            docs.iter()
                .map(|(id, text)| {
                    VectorRecord::new(id.clone(), Embedding(vec![0.1, 0.2, 0.3, 0.4]), text)
                })
                .collect(),
        )
        .await
        .unwrap();

    // Selective, common-only, mixed, and absent-term workloads.
    let queries = [
        "rare17",             // selective
        "rare101 rare250",    // multiple rares
        "alpha beta gamma",   // common-only (pruning can't skip — still exact)
        "rare17 alpha omega", // mixed: rare + commons → the pruning showcase
        "mid7 mid31",         // mid-frequency
        "alpha rare9999x",    // one absent term
        "zzz-not-a-word",     // fully absent
    ];
    for q in queries {
        let got = recall.lexical_search(q, 10).await.unwrap();
        let want = brute_topk(&docs, q, 10);
        assert_eq!(
            got.len(),
            want.len(),
            "query {q:?}: got {:?}",
            got.iter().map(|c| c.id.as_str()).collect::<Vec<_>>()
        );
        for (g, (wid, wscore)) in got.iter().zip(&want) {
            assert_eq!(g.id.as_str(), wid, "query {q:?} rank order");
            assert!(
                (g.score - wscore).abs() < 1e-3,
                "query {q:?} score {} vs {}",
                g.score,
                wscore
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn df_counters_track_writes_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "df").unwrap();
    let recall = estate.recall();
    let rec =
        |id: &str, text: &str| VectorRecord::new(id, Embedding(vec![0.1, 0.2, 0.3, 0.4]), text);

    recall
        .upsert(vec![
            rec("a", "quantum estate quantum"), // repeated term: df counts once
            rec("b", "quantum flows"),
            rec("c", "plain document"),
        ])
        .await
        .unwrap();
    assert_eq!(estate.term_df("quantum").unwrap(), 2);
    assert_eq!(estate.term_df("flows").unwrap(), 1);
    assert_eq!(estate.term_df("absent").unwrap(), 0);

    // Overwrite: a drops "quantum".
    recall
        .upsert(vec![rec("a", "different words entirely")])
        .await
        .unwrap();
    assert_eq!(estate.term_df("quantum").unwrap(), 1);

    // Remove: b goes too.
    recall.remove(&"b".into()).await.unwrap();
    assert_eq!(estate.term_df("quantum").unwrap(), 0);
    // Search agrees with the counters.
    assert!(recall
        .lexical_search("quantum", 5)
        .await
        .unwrap()
        .is_empty());
}
