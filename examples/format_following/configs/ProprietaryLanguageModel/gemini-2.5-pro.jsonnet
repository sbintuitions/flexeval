{
  class_path: 'LiteLLMChatAPI',
  init_args: {
    model: 'vertex_ai/gemini-2.5-pro',
    model_limit_completion_tokens: 8192,
    default_gen_kwargs: {
      max_tokens: 8192,
      temperature: 0.7,
      // Since thinking cannot be turned off
      thinking: { type : "enabled", budget_tokens: 128},
      top_p: 0.9,
      vertex_project: "your_vertex_project",
      vertex_location: "your_vertex_location",
      safety_settings: [
        {
            category: "HARM_CATEGORY_HARASSMENT",
            threshold: "BLOCK_NONE",
        },
        {
            category: "HARM_CATEGORY_HATE_SPEECH",
            threshold: "BLOCK_NONE",
        },
        {
            category: "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold: "BLOCK_NONE",
        },
        {
            category: "HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold: "BLOCK_NONE",
        },
        {
            category: "HARM_CATEGORY_CIVIC_INTEGRITY",
            threshold: "BLOCK_NONE",
        },
      ],
    },
  },
}
