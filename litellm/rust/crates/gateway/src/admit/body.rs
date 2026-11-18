//! Which request bodies the native `/v1/messages` answers the way the Python proxy does.
//!
//! The proxy reads its own parameters out of a request body (`user`, `tags`, `timeout`,
//! `num_retries`, `fallbacks` and more) and never sends them to the provider. The native route
//! sends every key it does not type on to the provider, which refuses it. So a body is admitted
//! only when each of its top-level keys is a parameter of an Anthropic Messages request.
//!
//! Judging the body means reading it before the request is admitted or refused. A refused request
//! still goes to Python with the bytes the client sent.

use std::collections::BTreeMap;

use axum::{
    body::{Body, Bytes},
    http::{header::CONTENT_ENCODING, request::Parts},
};
use futures_util::{Stream, StreamExt, future, stream};
use http_body_util::BodyExt;
use litellm_gateway_inference::MAX_BODY_BYTES;
use serde::de::IgnoredAny;

/// The top-level keys the native route types, `MessagesRequest` in `litellm-llms-types`. A key
/// missing here only sends its requests to Python.
const MESSAGES_PARAMS: [&str; 23] = [
    "model",
    "messages",
    "max_tokens",
    "system",
    "metadata",
    "stop_sequences",
    "stream",
    "temperature",
    "top_p",
    "top_k",
    "tools",
    "tool_choice",
    "thinking",
    "service_tier",
    "container",
    "mcp_servers",
    "context_management",
    "output_format",
    "output_config",
    "speed",
    "inference_geo",
    "reasoning_effort",
    "compaction",
];

/// A body that is a JSON object of Messages parameters and nothing else.
fn is_honored(body: &[u8]) -> bool {
    serde_json::from_slice::<BTreeMap<String, IgnoredAny>>(body).is_ok_and(|keys| {
        keys.keys()
            .all(|key| MESSAGES_PARAMS.contains(&key.as_str()))
    })
}

/// `read` followed by `rest`, as one body.
fn replayed(
    read: Vec<Bytes>,
    rest: impl Stream<Item = Result<Bytes, axum::Error>> + Send + 'static,
) -> Body {
    Body::from_stream(stream::iter(read).map(Ok).chain(rest))
}

/// `body` to send on, carrying the bytes the client sent, and whether the native route answers
/// it as the proxy would. A body that is compressed, or larger than the native route reads, is
/// left to Python without being judged, and one that breaks is sent on broken.
pub(super) async fn judged(request: &Parts, mut body: Body) -> (Body, bool) {
    if request.headers.contains_key(CONTENT_ENCODING) {
        return (body, false);
    }
    let mut read = Vec::new();
    let mut size = 0;
    while let Some(frame) = body.frame().await {
        let data = match frame.map(|frame| frame.into_data()) {
            Ok(Ok(data)) => data,
            Ok(Err(_trailers)) => continue,
            Err(broke) => {
                return (
                    replayed(read, stream::once(future::ready(Err(broke)))),
                    false,
                );
            }
        };
        size += data.len();
        read.push(data);
        if size > MAX_BODY_BYTES {
            return (replayed(read, body.into_data_stream()), false);
        }
    }
    let whole = Bytes::from(read.concat());
    let honored = is_honored(&whole);
    (Body::from(whole), honored)
}

#[cfg(test)]
mod tests {
    use axum::http::Request;
    use rstest::rstest;
    use serde_json::{Map, Value, json};

    use super::*;

    fn parts(encoding: Option<&str>) -> Parts {
        encoding
            .into_iter()
            .fold(Request::post("/v1/messages"), |request, encoding| {
                request.header(CONTENT_ENCODING, encoding)
            })
            .body(())
            .unwrap()
            .into_parts()
            .0
    }

    async fn sent_on(body: Body) -> Bytes {
        axum::body::to_bytes(body, usize::MAX).await.unwrap()
    }

    #[rstest]
    #[case::messages_params(
        r#"{"model":"claude","max_tokens":16,"stream":true,"messages":[]}"#,
        true
    )]
    #[case::a_proxy_param(r#"{"model":"claude","messages":[],"user":"end-user"}"#, false)]
    #[case::an_unknown_param(r#"{"model":"claude","messages":[],"not_a_param":1}"#, false)]
    #[case::not_an_object(r#"["model"]"#, false)]
    #[case::not_json("model=claude", false)]
    #[tokio::test]
    async fn a_body_of_messages_params_only_is_honored(
        #[case] sent: &'static str,
        #[case] expected: bool,
    ) {
        let (body, honored) = judged(&parts(None), Body::from(sent)).await;

        assert_eq!(
            (honored, sent_on(body).await),
            (expected, Bytes::from(sent))
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_compressed_body_is_left_to_python_unread() {
        let sent = r#"{"model":"claude","messages":[]}"#;

        let (body, honored) = judged(&parts(Some("gzip")), Body::from(sent)).await;

        assert_eq!((honored, sent_on(body).await), (false, Bytes::from(sent)));
    }

    #[rstest]
    #[tokio::test]
    async fn a_body_larger_than_the_native_route_reads_is_sent_on_whole() {
        let chunk = Bytes::from(vec![b' '; 1024 * 1024]);
        let chunks = MAX_BODY_BYTES / chunk.len() + 2;
        let sent = Body::from_stream(stream::iter(
            std::iter::repeat_n(chunk.clone(), chunks).map(Ok::<_, axum::Error>),
        ));

        let (body, honored) = judged(&parts(None), sent).await;

        assert_eq!(
            (honored, sent_on(body).await.len()),
            (false, chunks * chunk.len())
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_body_that_breaks_is_sent_on_broken() {
        let sent = Body::from_stream(stream::iter([
            Ok(Bytes::from_static(b"{\"model\":")),
            Err(std::io::Error::other("the client went away")),
        ]));

        let (body, honored) = judged(&parts(None), sent).await;

        assert!(!honored);
        assert!(axum::body::to_bytes(body, usize::MAX).await.is_err());
    }

    /// Every listed key is one the native route types, so none of them reaches the provider as
    /// an unknown parameter.
    #[rstest]
    fn every_listed_param_is_typed_by_the_native_route() {
        let body: Map<String, Value> = MESSAGES_PARAMS
            .iter()
            .map(|name| ((*name).to_owned(), Value::Null))
            .chain([
                ("model".to_owned(), json!("claude")),
                ("messages".to_owned(), json!([])),
            ])
            .collect();

        let request = litellm_inference_messages::messages_body(body).unwrap();

        assert_eq!(request.params.extra, Map::new());
    }
}
