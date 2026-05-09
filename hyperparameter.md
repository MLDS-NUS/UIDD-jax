\begin{table}[h]
\centering
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccc}
\hline
& Linear (D.3) & Polymer (D.4) & SGLD / ICA (D.5) & Nonlinear SDE (new) \\
\hline
\multicolumn{5}{l}{\emph{Potential network \(V_\theta\)}}\\
\quad Form of \(V_\theta\) & scaled MLP potential & residual quadratic \(\alpha\|Z\|^2 + \frac12\|\phi(Z)+\Gamma Z\|^2\times \mathrm{scale}\) & residual quadratic \(\alpha\|x\|^2 + \|\phi(x)/\mathrm{scale}+\Gamma x\|^2\times \mathrm{scale}\) & scaled MLP potential \\
\quad Hidden widths of \(U\) & \((64,32)\) & \((128)\) & \((128)\) & \((64,32)\) \\
\quad Activation & ReCU & shifted ReQU & tanh & ReCU \\
\quad Embedding dimension \(m\) & --- & \(32\) & \(32\) & --- \\
\quad \(\beta_1\) & --- & --- & --- & --- \\
\quad \(\beta_2\) & --- & --- & --- & --- \\
\quad \(\beta_3\) & --- & --- & --- & --- \\
\hline
\multicolumn{5}{l}{\emph{Antisymmetric network \(H\) (\(W_\theta=\sum_d H_d J_d\))}}\\
\quad Hidden widths & \((32,16)\) & \((128,128)\) & \((128,128)\) & \((32,16)\) \\
\quad Activation & ReCU & tanh & tanh & ReCU \\
\hline
\multicolumn{5}{l}{\emph{Diffusion \(\sigma_\theta\)}}\\
\quad Form & state-indep.\ trainable lower-triangular & state-indep.\ diagonal & state-dep.\ trainable lower-triangular & state-indep.\ trainable lower-triangular \\
\quad Hidden widths of \(\sigma_1,\sigma_2\) & --- & --- & \((32,32)\) each & --- \\
\quad Activation & --- & --- & tanh & --- \\
\hline
\multicolumn{5}{l}{\emph{Training}}\\
\quad Optimizer & Adam & Adam & Adam & Adam \\
\quad Learning rate & \(1\times10^{-3}\) & \(1\times10^{-3}\) & \(1\times10^{-3}\) & \(1\times10^{-3}\) \\
\quad Batch size & \(5\times 10^{4}\) & \(2\) trajectories & \(1\times 10^{5}\) & \(1\times 10^{5}\) \\
\quad Number of epochs & \(5\,000\) & \(2\,000\) & \(500\) & \(3\,000\) main; \(5\,000\) in sensitivity sweep \\
\quad Random seeds & \(\{0,1,10,20\}\) across runs & \(\{0,1,12,123\}\) & \(\{10,20,30,40\}\) & \(\{0,1,12,123,1234\}\) \\
\hline
\multicolumn{5}{l}{\emph{Data and integration}}\\
\quad Training trajectories & \(9\,000\) & \(80\%\) of each loaded dataset & \(160\,000\) generated per cache & \(9\,000\) \\
\quad Test trajectories & \(1\,000\) & \(20\%\) of each loaded dataset & \(40\,000\) generated per cache & \(1\,000\) \\
\quad Observation \(\Delta t\) & \(0.01\) & dataset-provided & \(0.05\) & \(0.01\) main; \(0.005\) also run in \texttt{train.sh} \\
\quad Integration step & \(0.01\) & \(5\times 10^{-4}\) for learned-model simulation & \(0.05\) discrete update & \(0.001\) main; \(0.0005\) also run in \texttt{train.sh} \\
\hline
\multicolumn{5}{l}{\emph{Auxiliary settings}}\\
\quad PCA--ResNet \(\lambda\) (line 1014) & --- & --- & --- & --- \\
\quad Closure dimensionality (\(Z_1,Z_2,Z_3\)) & --- & \(3\) & --- & --- \\
\hline
\end{tabular}
}
\caption{Hyperparameters matched to the current runtime files in the repository. Where runtime behavior depends on external cached datasets or multiple launch scripts, the table records the actual code path used by those scripts rather than only the YAML defaults. ``---'' indicates not applicable.}
\label{tab:hyperparameters}
\end{table}