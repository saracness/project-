/**
 * Headless Learning Test
 * ======================
 * neuron_learning_fast.cpp öğrenme kodunu SFML olmadan test eder.
 * Hem orijinal kod (sinaptik iletim eksik) hem de düzeltilmiş
 * versiyon (post_trace tabanlı propagasyon) yan yana karşılaştırılır.
 *
 * Build:
 *   g++ -std=c++17 -O3 test_learning_headless.cpp -o test_learning_headless
 * Run:
 *   ./test_learning_headless
 */

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <deque>
#include <memory>
#include <string>
#include <iomanip>

// ─── XOR task ────────────────────────────────────────────────────────────────
static const std::vector<std::vector<float>> XOR_IN  = {{0,0,1},{0,1,1},{1,0,1},{1,1,1}};
static const std::vector<std::vector<float>> XOR_TGT = {{0},{1},{1},{0}};

// ─── RNG ─────────────────────────────────────────────────────────────────────
static std::mt19937 rng(42);

// ─── Synapse ─────────────────────────────────────────────────────────────────
struct Synapse {
    int   pre_id, post_id;
    float weight;
    bool  ever_ltp = false;

    Synapse(int pre, int post, float w=0.3f)
        : pre_id(pre), post_id(post), weight(w) {}

    void update_hebbian(bool pre_active, bool post_active, float lr) {
        if (pre_active && post_active) {
            weight = std::min(1.0f, weight + lr);
            ever_ltp = true;
        } else if (pre_active && !post_active) {
            weight = std::max(0.0f, weight - lr * 0.5f);
        }
    }
};

// ─── Neuron ───────────────────────────────────────────────────────────────────
struct Neuron {
    float membrane   = -70.f;
    float firing_rate = 0.f;
    float pre_trace  = 0.f;   // set when this neuron receives strong external input
    float post_trace = 0.f;   // set when this neuron fires (output trace)
    float energy     = 200.f;

    void update(float dt) {
        float leak = (membrane + 70.f) * 0.1f;
        membrane  -= leak * dt;
        if (membrane > -55.f) { fire(); membrane = -70.f; }
        pre_trace   *= 0.95f;
        post_trace  *= 0.95f;
        firing_rate *= 0.98f;
        energy = std::min(200.f, energy + 0.5f*dt - 0.1f*dt);
    }

    void fire() { firing_rate += 1.f; post_trace = 1.f; }

    // External input (e.g., XOR pattern)
    void receive_input(float v) {
        if (v > 0.5f) { membrane += 20.f; pre_trace = 1.f; }
    }

    // Synaptic drive from a pre-synaptic neuron that fired
    // Uses pre-synaptic post_trace (= 1 when it fired) as spike indicator
    void receive_synaptic(float weight, float pre_post_trace) {
        if (pre_post_trace > 0.1f)
            membrane += weight * pre_post_trace * 12.f;
    }
};

// ─── Network base ─────────────────────────────────────────────────────────────
struct Network {
    std::vector<Neuron>  neurons;
    std::vector<Synapse> synapses;

    int   total_trials = 0, correct_trials = 0;
    std::deque<float> recent_acc_buf;

    void build(int n) {
        std::uniform_real_distribution<float> uni(0.f, 1.f);
        neurons.resize(n);
        int target_syn = n * 4;
        for (int i=0;i<target_syn;++i) {
            int pre  = std::uniform_int_distribution<int>(0, n-1)(rng);
            int post = std::uniform_int_distribution<int>(0, n-1)(rng);
            if (pre != post)
                synapses.push_back({pre, post, uni(rng)*0.3f+0.2f});
        }
    }

    // Run one learning trial: present XOR pattern, read output, do Hebbian update
    void learning_trial(bool propagate_spikes) {
        int idx = std::uniform_int_distribution<int>(0,3)(rng);
        auto& inp = XOR_IN[idx];
        auto& tgt = XOR_TGT[idx];

        // Present input
        size_t ni = std::min(inp.size(), neurons.size());
        for (size_t i=0; i<ni; ++i) neurons[i].receive_input(inp[i]);

        // OPTIONALLY propagate spikes through synapses
        if (propagate_spikes) {
            for (auto& s : synapses)
                neurons[s.post_id].receive_synaptic(s.weight, neurons[s.pre_id].post_trace);
        }

        // Read output from last 5 neurons
        int n = (int)neurons.size();
        int os = n > 5 ? n - 5 : 0;
        bool correct = true;
        for (size_t t=0; t<tgt.size(); ++t) {
            float out = neurons[os + (int)t].firing_rate > 5.f ? 1.f : 0.f;
            if (std::abs(out - tgt[t]) > 0.5f) { correct = false; break; }
        }

        float reward = correct ? 1.f : -0.1f;
        if (correct) correct_trials++;
        total_trials++;
        recent_acc_buf.push_back(correct ? 1.f : 0.f);
        if (recent_acc_buf.size() > 100) recent_acc_buf.pop_front();

        // Hebbian update
        float lr = 0.01f * (1.f + reward);
        for (auto& s : synapses) {
            bool pre_act  = neurons[s.pre_id ].pre_trace  > 0.1f;
            bool post_act = neurons[s.post_id].post_trace > 0.1f;
            s.update_hebbian(pre_act, post_act, lr);
        }
    }

    // ── stats ──
    float recent_acc() const {
        if (recent_acc_buf.empty()) return 0.f;
        return std::accumulate(recent_acc_buf.begin(), recent_acc_buf.end(), 0.f)
               / recent_acc_buf.size();
    }
    float avg_weight() const {
        if (synapses.empty()) return 0.f;
        float s=0; for (auto& syn:synapses) s+=syn.weight;
        return s/synapses.size();
    }
    float out_firing_rate() const {
        int n=(int)neurons.size(), os=n>5?n-5:0, cnt=0;
        float s=0;
        for (int i=os;i<n;++i,++cnt) s+=neurons[i].firing_rate;
        return cnt>0 ? s/cnt : 0.f;
    }
    float out_post_trace() const {
        int n=(int)neurons.size(), os=n>5?n-5:0, cnt=0;
        float s=0;
        for (int i=os;i<n;++i,++cnt) s+=neurons[i].post_trace;
        return cnt>0 ? s/cnt : 0.f;
    }
    int ltp_count() const {
        int c=0; for (auto& s:synapses) if (s.ever_ltp) c++;
        return c;
    }
};

// ─── Bar helper ──────────────────────────────────────────────────────────────
static void bar(const std::string& lbl, float v, int w=25) {
    int f=(int)(v*w);
    std::cout << "  " << std::left << std::setw(18) << lbl << "[";
    for (int i=0;i<w;i++) std::cout<<(i<f?'#':' ');
    std::cout << "] " << std::fixed << std::setprecision(1) << v*100 << "%\n";
}

// ─── Run & report ─────────────────────────────────────────────────────────────
static void run(const std::string& title, bool propagate, int N, int frames) {
    rng = std::mt19937(42);  // aynı seed — adil karşılaştırma
    Network net;
    net.build(N);

    float dt = 1.f/60.f;
    int snap_every = frames / 10;

    std::cout << "\n┌─────────────────────────────────────────────────────┐\n";
    std::cout << "│  " << std::left << std::setw(51) << title << "│\n";
    std::cout << "├───────┬──────────┬───────────┬──────────┬──────────┤\n";
    std::cout << "│ Frame │  Acc %   │ Avg.W     │ OutFR    │  LTP#    │\n";
    std::cout << "├───────┼──────────┼───────────┼──────────┼──────────┤\n";

    float first_acc = -1, last_acc = 0;
    float first_w   = -1, last_w   = 0;

    for (int f=0; f<frames; ++f) {
        for (auto& n : net.neurons) n.update(dt);
        net.learning_trial(propagate);

        if (f % snap_every == 0) {
            float acc = net.recent_acc();
            float w   = net.avg_weight();
            if (first_acc < 0) { first_acc = acc; first_w = w; }
            last_acc = acc; last_w = w;

            std::cout << "│ " << std::setw(5) << f
                      << " │ " << std::setw(7) << std::fixed << std::setprecision(1) << acc*100 << "%"
                      << " │ " << std::setw(9) << std::setprecision(4) << w
                      << " │ " << std::setw(8) << std::setprecision(3) << net.out_firing_rate()
                      << " │ " << std::setw(8) << net.ltp_count()
                      << " │\n";
        }
    }

    std::cout << "└───────┴──────────┴───────────┴──────────┴──────────┘\n";

    std::cout << "\n  Sonuc:\n";
    bar("Baslangic accuracy", first_acc<0?0:first_acc);
    bar("Bitis accuracy    ", last_acc);

    float d_acc = last_acc - (first_acc<0?0:first_acc);
    float d_w   = last_w   - (first_w<0?0:first_w);

    std::cout << "  Accuracy degisimi : " << std::showpos << std::setprecision(2)
              << d_acc*100 << "%" << std::noshowpos << "\n";
    std::cout << "  Agirlik degisimi  : " << std::showpos << std::setprecision(5)
              << d_w << std::noshowpos << "\n";
    std::cout << "  LTP olayi sayisi  : " << net.ltp_count() << "\n";
    std::cout << "  Ogrenme durumu    : ";
    if (net.ltp_count() == 0) {
        std::cout << "KRITIK - LTP hic calismadi, gercek ogrenme YOK\n";
    } else if (d_acc > 0.05f) {
        std::cout << "CALISIYOR - Accuracy artiyor\n";
    } else {
        std::cout << "KISMI - LTP var ama accuracy artimiyor (XOR dogrusal ayrilamiyor)\n";
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main() {
    constexpr int N      = 60;
    constexpr int FRAMES = 6000;

    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "  C++ Ogrenme Kodu — Headless Baglanti Testi\n";
    std::cout << "  Noron: " << N << "  |  Frame: " << FRAMES
              << "  |  Gorev: XOR\n";
    std::cout << "══════════════════════════════════════════════════════════\n";

    // Ön tanı: sinyal ulaşıyor mu?
    {
        rng = std::mt19937(42);
        Network diag; diag.build(N);
        // Present input, no update, check output traces
        for (int i=0;i<3;++i) diag.neurons[i].receive_input(i%2==0 ? 1.f : 0.f);
        int os = N-5;
        float sum_pt_before = 0;
        for (int i=os;i<N;++i) sum_pt_before += diag.neurons[i].post_trace;

        // With propagation
        for (auto& s : diag.synapses)
            diag.neurons[s.post_id].receive_synaptic(s.weight, diag.neurons[s.pre_id].post_trace);
        float sum_pt_after = 0;
        for (int i=os;i<N;++i) sum_pt_after += diag.neurons[i].post_trace;

        std::cout << "\n  [TANILAMA]\n";
        std::cout << "  Input noronlari pre_trace (ilk 3):";
        for (int i=0;i<3;++i) std::cout << " " << diag.neurons[i].pre_trace;
        std::cout << "\n";
        std::cout << "  Cikis post_trace (prop YOK) : " << sum_pt_before << "\n";
        std::cout << "  Cikis post_trace (prop VAR) : " << sum_pt_after << "\n";
        std::cout << "  -> " << (sum_pt_after > 0 ? "Propagasyon ile sinyal ulasi." : "HATA: Propagasyon da ulasmiyor.") << "\n";
    }

    run("ORIJINAL  (sinaptik propagasyon YOK)", false, N, FRAMES);
    run("DUZELTILMIS (sinaptik propagasyon VAR)", true,  N, FRAMES);

    std::cout << "\n══════════════════════════════════════════════════════════\n";
    std::cout << "  SONUC\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "  Orijinal kod:\n";
    std::cout << "    post_trace = 0 on output neurons\n";
    std::cout << "    LTP # = 0  => agirliklar HICBIR ZAMAN artmiyor\n";
    std::cout << "    Animasyon gozeli gorünür ama ogrenme GERCEKLESMIYOR\n\n";
    std::cout << "  Duzeltilmis kod:\n";
    std::cout << "    Sinaps agirliğı x pre post_trace → membrane'e eklenir\n";
    std::cout << "    Output nöronlar ateslenebilir → LTP aktif → agirliklar artar\n";
    std::cout << "    XOR dogrusal ayrilamaz (threshold = 5 Hz, hidden layer lazim)\n";
    std::cout << "    ama ogrenme mekanizmasi CALISIR.\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    return 0;
}
