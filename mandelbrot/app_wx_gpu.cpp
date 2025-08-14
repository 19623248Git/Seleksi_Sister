#include <wx/wx.h>
#include <wx/rawbmp.h>
#include <vector>
#include <string>

#include "src/wx-gpu-engine/mandelbrot_renderer_gpu.hpp"
#include "src/wx-engine/mandelbrot_engine.hpp"

class MandelbrotCanvas : public wxPanel
{
public:
    MandelbrotCanvas(wxWindow* parent) : wxPanel(parent)
    {
        this->Bind(wxEVT_PAINT, &MandelbrotCanvas::OnPaint, this);
        this->Bind(wxEVT_LEFT_DOWN, &MandelbrotCanvas::OnMouseDown, this);
        this->Bind(wxEVT_RIGHT_DOWN, &MandelbrotCanvas::OnMouseDown, this);
        this->Bind(wxEVT_MOUSEWHEEL, &MandelbrotCanvas::OnMouseWheel, this);
        this->Bind(wxEVT_MIDDLE_DOWN, &MandelbrotCanvas::OnPanStart, this);
        this->Bind(wxEVT_MIDDLE_UP, &MandelbrotCanvas::OnPanEnd, this);
        this->Bind(wxEVT_MOTION, &MandelbrotCanvas::OnMouseMove, this);
        this->Bind(wxEVT_LEAVE_WINDOW, &MandelbrotCanvas::OnLeaveWindow, this);
        this->SetBackgroundStyle(wxBG_STYLE_PAINT);
    }

    void UpdateSettingsAndRender(int iterations)
    {
        engine.setMaxIterations(iterations);
        Refresh();
    }

private:
    MandelbrotEngine engine;
    double centerX = -0.75;
    double centerY = 0.0;
    double scale = 0.004;
    bool isPanning = false;
    wxPoint panStartPoint;

    void OnPaint(wxPaintEvent& event)
    {
        wxPaintDC dc(this);
        wxSize clientSize = GetClientSize();
        int width = clientSize.GetWidth();
        int height = clientSize.GetHeight();
        if (width <= 0 || height <= 0) return;

        std::vector<Color> pixel_colors(width * height);
        double half_width_scaled = (width / 2.0) * scale;
        double half_height_scaled = (height / 2.0) * scale;
        double minReal = centerX - half_width_scaled;
        double maxReal = centerX + half_width_scaled;
        double minImag = centerY - half_height_scaled;
        double maxImag = centerY + half_height_scaled;

        renderWithCuda(pixel_colors, width, height, minReal, maxReal, minImag, maxImag, engine.getMaxIterations());

        wxBitmap bmp(width, height, 24);
        wxNativePixelData pixelData(bmp);
        wxNativePixelData::Iterator p(pixelData);
        for (int y = 0; y < height; ++y) {
            wxNativePixelData::Iterator rowStart = p;
            for (int x = 0; x < width; ++x) {
                const Color& color = pixel_colors[y * width + x];
                p.Red() = color.r;
                p.Green() = color.g;
                p.Blue() = color.b;
                p++;
            }
            p = rowStart;
            p.OffsetY(pixelData, 1);
        }
        dc.DrawBitmap(bmp, 0, 0);
    }

    // --- ADDED: All the missing event handler methods ---
    void OnMouseDown(wxMouseEvent& event) {
        wxPoint clickPos = event.GetPosition();
        wxSize clientSize = GetClientSize();
        centerX += (clickPos.x - clientSize.GetWidth() / 2.0) * scale;
        centerY += (clickPos.y - clientSize.GetHeight() / 2.0) * scale;
        if (event.LeftDown()) { scale /= 2.0; } 
        else if (event.RightDown()) { scale *= 2.0; }
        Refresh();
    }
    
    void OnMouseWheel(wxMouseEvent& event) {
        if (event.GetWheelRotation() == 0) return;
        wxPoint mousePos = event.GetPosition();
        wxSize clientSize = GetClientSize();
        int width = clientSize.GetWidth();
        int height = clientSize.GetHeight();
        double mouseReal_before = centerX + (mousePos.x - width / 2.0) * scale;
        double mouseImag_before = centerY + (mousePos.y - height / 2.0) * scale;
        double zoomFactor = 1.5;
        if (event.GetWheelRotation() > 0) { scale /= zoomFactor; } 
        else { scale *= zoomFactor; }
        centerX = mouseReal_before - (mousePos.x - width / 2.0) * scale;
        centerY = mouseImag_before - (mousePos.y - height / 2.0) * scale;
        Refresh();
    }

    void OnPanStart(wxMouseEvent& event) { isPanning = true; panStartPoint = event.GetPosition(); }
    void OnPanEnd(wxMouseEvent& event) { isPanning = false; }
    void OnLeaveWindow(wxMouseEvent& event) { isPanning = false; }

    void OnMouseMove(wxMouseEvent& event) {
        if (!isPanning || !event.MiddleIsDown()) return;
        wxPoint currentPoint = event.GetPosition();
        wxPoint delta = currentPoint - panStartPoint;
        centerX -= delta.x * scale;
        centerY -= delta.y * scale;
        panStartPoint = currentPoint;
        Refresh();
    }
};

class MandelbrotFrame : public wxFrame
{
public:
    MandelbrotFrame(const wxString& title, const wxSize& size)
        : wxFrame(NULL, wxID_ANY, title, wxDefaultPosition, size)
    {
        wxPanel* controlPanel = new wxPanel(this);
        iterCtrl = new wxTextCtrl(controlPanel, wxID_ANY, "255");
        wxButton* applyButton = new wxButton(controlPanel, wxID_ANY, "Apply Settings");
        canvas = new MandelbrotCanvas(this);

        wxBoxSizer* mainSizer = new wxBoxSizer(wxHORIZONTAL);
        wxBoxSizer* controlSizer = new wxBoxSizer(wxVERTICAL);
        controlSizer->Add(new wxStaticText(controlPanel, wxID_ANY, "Max Iterations:"), 0, wxALL, 5);
        controlSizer->Add(iterCtrl, 0, wxEXPAND | wxALL, 5);
        controlSizer->Add(applyButton, 0, wxEXPAND | wxALL, 5);
        controlPanel->SetSizer(controlSizer);
        mainSizer->Add(controlPanel, 0, wxEXPAND | wxALL, 10);
        mainSizer->Add(canvas, 1, wxEXPAND);
        this->SetSizerAndFit(mainSizer);
        this->SetMinSize(wxSize(600, 400));

        // Use a lambda for the button event to keep it self-contained
        applyButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event){
            long iterations;
            if (iterCtrl->GetValue().ToLong(&iterations) && iterations > 0) {
                canvas->UpdateSettingsAndRender(static_cast<int>(iterations));
            } else {
                wxMessageBox("Please enter a valid positive number for iterations.", "Invalid Input", wxOK | wxICON_ERROR);
            }
        });

        // Initial render
        canvas->UpdateSettingsAndRender(255);
    }
private:
    MandelbrotCanvas* canvas;
    wxTextCtrl* iterCtrl;
};

class MandelbrotApp : public wxApp
{
public:
    virtual bool OnInit()
    {
        MandelbrotFrame* frame = new MandelbrotFrame("Interactive Mandelbrot Explorer (GPU 🚀)", wxSize(1280, 720));
        frame->Show(true);
        frame->Center();
        return true;
    }
};

wxIMPLEMENT_APP(MandelbrotApp);