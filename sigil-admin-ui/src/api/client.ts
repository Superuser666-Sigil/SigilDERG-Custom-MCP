export interface AdminClientOptions {
  baseUrl?: string;
  apiKey?: string | null;
}

export class AdminClient {
  private baseUrl: string;
  private apiKey: string | null;

  constructor(options: AdminClientOptions = {}) {
    this.baseUrl = options.baseUrl ?? "http://127.0.0.1:8765";
    this.apiKey = options.apiKey ?? null;
  }

  setApiKey(key: string | null) {
    this.apiKey = key;
  }

  private headers(extra?: Record<string, string>) {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(extra ?? {}),
    };
    if (this.apiKey) {
      headers["X-Admin-Key"] = this.apiKey;
    }
    return headers;
  }

  private async handle<T>(resp: Response): Promise<T> {
    const text = await resp.text();
    const data = text ? JSON.parse(text) : null;

    if (!resp.ok) {
      const error = (data && (data.error || data.detail)) || resp.statusText;
      throw new Error(error);
    }
    return data as T;
  }

  async get<T>(path: string): Promise<T> {
    const resp = await fetch(this.baseUrl + path, {
      method: "GET",
      headers: this.headers(),
    });
    return this.handle<T>(resp);
  }

  async post<T>(path: string, body: unknown): Promise<T> {
    const resp = await fetch(this.baseUrl + path, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(body),
    });
    return this.handle<T>(resp);
  }
}
