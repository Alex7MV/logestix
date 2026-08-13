// StreamClient — incremental SSE ingestion over fetch + ReadableStream.
//
// The raw SSE byte stream is decoded in-place, framed on blank lines, and each
// `data:` payload's `choices[0].delta.content` is forwarded synchronously to the
// sink. Frames are consumed exactly once; there is no re-buffering of the whole
// response and no per-event DOM reconstruction (see renderer.append).

export class StreamClient {
  /**
   * @param {string} endpoint base URL, e.g. "/v1/chat/completions"
   */
  constructor(endpoint = '/v1/chat/completions') {
    this.endpoint = endpoint;
    this.abort = new AbortController();
  }

  cancel() {
    this.abort.abort();
  }

  /**
   * @param {object} payload OpenAI request body ({ model, messages, stream: true })
   * @param {(delta: string) => void} onDelta
   * @returns {Promise<void>} resolves on [DONE] or when the stream closes.
   */
  async stream(payload, onDelta) {
    const resp = await fetch(this.endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify({ ...payload, stream: true }),
      signal: this.abort.signal,
    });
    if (!resp.ok || !resp.body) {
      throw new Error(`stream failed: HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    // Decode the transport into a text frame buffer, then split on blank-line
    // SSE delimiters. Each completed event is processed and released.
    for (;;) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: true });

      let sep;
      while ((sep = buffer.indexOf('\n\n')) >= 0) {
        const event = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        for (const line of event.split('\n')) {
          if (!line.startsWith('data:')) continue;
          const data = line.slice(5).trim();
          if (data === '[DONE]') return;
          if (data === '') continue;
          const json = JSON.parse(data);
          const delta = json.choices?.[0]?.delta?.content;
          if (delta) onDelta(delta);
        }
      }
      if (done) break;
    }
  }
}
