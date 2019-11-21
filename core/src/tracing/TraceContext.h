#pragma once

#include <opentracing/tracer.h>
#include <string>

class TraceContext {
 public:
    explicit TraceContext(std::unique_ptr<opentracing::Span>& span);

    std::unique_ptr<TraceContext>
    Child(const std::string& operation_name) const;

    std::unique_ptr<TraceContext>
    Follower(const std::string& operation_name) const;

    const std::unique_ptr<opentracing::Span>&
    getSpan() const;

 private:
    //    std::unique_ptr<opentracing::SpanContext> span_context_;
    std::unique_ptr<opentracing::Span> span_;
};